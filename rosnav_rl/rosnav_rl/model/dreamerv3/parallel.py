import atexit
import enum
import os
import sys
import time
import traceback
from functools import partial as bind
from typing import Any, Callable, Collection

import cloudpickle


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)


class Parallel:
    def __init__(self, ctor, strategy):
        """
        Initialize the parallel execution class.

        Args:
            ctor: A constructor function that creates the object to be executed in parallel.
            strategy: A strategy object that determines how parallel execution should be handled.

        Attributes:
            worker: A Worker instance that handles parallel execution.
            callables: A dictionary that stores callable methods.
            __ctor: Private attribute to store the constructor reference.
        """

        self.worker = Worker(bind(self._respond, ctor), strategy, state=True)
        self.callables = {}
        self.__ctor = None

    def __getattr__(self, name):
        """
        Override attribute access for the worker proxy to handle remote method calls and attribute reads.

        This method is called when a non-existent attribute is accessed on the object. It handles three cases:
        1. Protected/private attributes (starting with '_')
        2. Callable attributes that can be executed remotely
        3. Regular attributes that need to be read from the remote worker

        Args:
            name (str): The name of the attribute being accessed

        Returns:
            Union[Callable, Any]: Either a bound method proxy for callable attributes,
                                 or the value for regular attributes

        Raises:
            AttributeError: If accessing a protected/private attribute
            ValueError: If the attribute doesn't exist on the remote worker

        Notes:
            - Caches callable status in self.callables dictionary
            - Uses PMessage enum to communicate with remote worker
        """
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            if name not in self.callables:
                self.callables[name] = self.worker(PMessage.CALLABLE, name)()
            if self.callables[name]:
                return bind(self.worker, PMessage.CALL, name)
            else:
                return self.worker(PMessage.READ, name)()
        except AttributeError:
            raise ValueError(name)

    def __len__(self):
        return self.worker(PMessage.CALL, "__len__")()

    def close(self):
        self.worker.close()

    @staticmethod
    def _respond(ctor, state, message, name, *args, **kwargs):
        """Process messages related to instance methods and attributes.

        Args:
            ctor: The constructor or class to instantiate if state is None
            state: The current instance state, will be set to ctor if None
            message: The message type from PMessage enum indicating operation to perform
            name: String name of the method or attribute to access
            *args: Variable length argument list to pass to called methods
            **kwargs: Arbitrary keyword arguments to pass to called methods

        Returns:
            tuple: (state, result) where:
                - state: The instance state (unchanged from input)
                - result: The result of the requested operation:
                    - For CALLABLE: Boolean indicating if attribute is callable
                    - For CALL: Result of calling the named method
                    - For READ: Value of the named attribute

        Raises:
            AssertionError: If args/kwargs provided for CALLABLE or READ messages
        """
        if state is None:
            state = ctor()

        state = state or ctor
        if message == PMessage.CALLABLE:
            assert not args and not kwargs, (args, kwargs)
            result = callable(getattr(state, name))
        elif message == PMessage.CALL:
            result = getattr(state, name)(*args, **kwargs)
        elif message == PMessage.READ:
            assert not args and not kwargs, (args, kwargs)
            result = getattr(state, name)
        return state, result


class PMessage(enum.Enum):
    """Enumeration class for process message types.

    This enum defines different types of messages that can be passed between processes:

    Attributes:
        CALLABLE (2): Represents a callable object or function message.
        CALL (3): Represents a call operation message.
        READ (4): Represents a read operation message.
    """

    CALLABLE = 2
    CALL = 3
    READ = 4


class Worker:
    initializers = []

    def __init__(self, fn, strategy="thread", state=False):
        """
        Initialize a parallel worker.

        This class allows for running a function in parallel, using different parallelization strategies.

        Args:
            fn (callable): The function to run in parallel.
            strategy (str, optional): The parallelization strategy. Can be "process", "daemon", or "thread".
                                      Defaults to "thread".
            state (bool, optional): If True, the function should accept state as its first argument.
                                   If False, the function will be wrapped to accept and return state.
                                   Defaults to False.

        Note:
            When state=False, the function is transformed to take a state argument and return a tuple
            of (state, function_result).
        """

        if not state:
            fn = lambda s, *args, fn=fn, **kwargs: (s, fn(*args, **kwargs))
        inits = self.initializers
        self.impl = {
            "process": bind(ProcessPipeWorker, initializers=inits),
            "daemon": bind(ProcessPipeWorker, initializers=inits, daemon=True),
        }[strategy](fn)
        self.promise = None

    def __call__(self, *args, **kwargs):
        """
        Call method for handling promise-based asynchronous operations.

        This method ensures proper handling of promised operations by:
        1. Checking and raising any previous exceptions
        2. Creating a new promise with the given arguments
        3. Returning the newly created promise

        Args:
            *args: Variable length argument list to be passed to the implementation
            **kwargs: Arbitrary keyword arguments to be passed to the implementation

        Returns:
            Promise: The newly created promise object from the implementation

        Raises:
            Any exception that might have been stored in the previous promise
        """
        self.promise and self.promise()  # Raise previous exception if any.
        self.promise = self.impl(*args, **kwargs)
        return self.promise

    def wait(self):
        return self.impl.wait()

    def close(self):
        self.impl.close()


class ProcessPipeWorker:
    def __init__(self, fn, initializers=(), daemon=False):
        """Initialize a parallel process for executing functions.

        This constructor sets up a parallel processing environment using multiprocessing,
        allowing functions to be executed in a separate process.

        Args:
            fn (callable): The function to be executed in parallel.
            initializers (tuple, optional): Tuple of initializer functions to be called
                before the main function. Defaults to ().
            daemon (bool, optional): If True, the process will be daemonic. Defaults to False.

        Attributes:
            _context: Multiprocessing context using 'spawn' method
            _pipe: Pipe connection for inter-process communication
            _process: Process object running the parallel execution
            _nextid: Counter for tracking message IDs
            _results: Dictionary storing results of parallel executions

        Raises:
            AssertionError: If initial communication check fails

        Note:
            Uses cloudpickle for serialization of functions and initializers.
            Automatically registers cleanup on program exit.
        """
        import multiprocessing

        import cloudpickle

        # if start_method is None:
        # Fork is not a thread safe method (see issue #217)
        # but is more user friendly (does not require to wrap the code in
        # a `if __name__ == "__main__":`)
        forkserver_available = "forkserver" in multiprocessing.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"

        self._context = multiprocessing.get_context(start_method)
        self._pipe, pipe = self._context.Pipe()
        fn = cloudpickle.dumps(fn)
        initializers = cloudpickle.dumps(initializers)
        self._process = self._context.Process(
            target=self._loop, args=(pipe, fn, initializers), daemon=daemon
        )
        self._process.start()
        self._nextid = 0
        self._results = {}
        assert self._submit(Message.OK)()
        atexit.register(self.close)

    def __call__(self, *args, **kwargs):
        """
        Call the wrapped function with the provided arguments.

        This method submits a RUN message to execute the wrapped function in parallel
        with the given arguments.

        Args:
            *args: Variable length argument list to pass to the wrapped function
            **kwargs: Arbitrary keyword arguments to pass to the wrapped function

        Returns:
            The result of the wrapped function call after execution

        Example:
            >>> parallel_fn = ParallelFunction(some_function)
            >>> result = parallel_fn(arg1, arg2, kwarg1=value1)
        """
        return self._submit(Message.RUN, (args, kwargs))

    def wait(self):
        pass

    def close(self):
        try:
            self._pipe.send((Message.STOP, self._nextid, None))
            self._pipe.close()
        except (AttributeError, IOError):
            pass  # The connection was already closed.
        try:
            self._process.join(0.1)
            if self._process.exitcode is None:
                try:
                    os.kill(self._process.pid, 9)
                    time.sleep(0.1)
                except Exception:
                    pass
        except (AttributeError, AssertionError):
            pass

    def _submit(self, message, payload=None):
        """
        Submit a message to the pipe with an optional payload and return a Future object.

        This method assigns a unique call ID to each message, sends the message through
        the pipe along with the ID and payload, and returns a Future object that can be
        used to retrieve the result of the operation.

        Args:
            message: The message to be sent through the pipe.
            payload (optional): Additional data to be sent along with the message.
                               Defaults to None.

        Returns:
            Future: A Future object that can be used to retrieve the result of the
                   operation associated with this message.

        Note:
            The Future object's result can be obtained by calling its result() method,
            which will block until the operation completes.
        """
        callid = self._nextid
        self._nextid += 1
        self._pipe.send((message, callid, payload))
        return Future(self._receive, callid)

    def _receive(self, callid):
        while callid not in self._results:
            try:
                message, callid, payload = self._pipe.recv()
            except (OSError, EOFError):
                raise RuntimeError("Lost connection to worker.")
            if message == Message.ERROR:
                raise Exception(payload)
            assert message == Message.RESULT, message
            self._results[callid] = payload
        return self._results.pop(callid)

    @staticmethod
    def _loop(pipe, function, initializers):
        """
        Internal worker loop for parallel processing that handles message-based communication.

        Args:
            pipe (multiprocessing.Connection): Pipe connection for inter-process communication
            function (bytes): Cloudpickle serialized function to be executed
            initializers (bytes): Cloudpickle serialized list of initialization functions

        The loop continuously:
            1. Deserializes the function and initializers
            2. Executes initializer functions
            3. Listens for messages through the pipe
            4. Processes different message types:
                - OK: Acknowledges with success
                - STOP: Terminates the loop
                - RUN: Executes the function with provided arguments
            5. Handles errors and sends error messages back through the pipe

        The function maintains a state between calls and supports keyboard interrupts.
        All communication is done through a bidirectional pipe using predefined message types.

        Raises:
            KeyError: If an invalid message type is received
            Exception: Catches and reports any other exceptions through the pipe

        Note:
            This is an internal method and should not be called directly.
        """
        callid = None
        state = None

        try:
            import cloudpickle

            initializers = cloudpickle.loads(initializers)
            function = cloudpickle.loads(function)

            [fn() for fn in initializers]
            while True:
                if not pipe.poll(0.1):
                    continue  # Wake up for keyboard interrupts.
                message, callid, payload = pipe.recv()
                if message == Message.OK:
                    pipe.send((Message.RESULT, callid, True))
                elif message == Message.STOP:
                    return
                elif message == Message.RUN:
                    args, kwargs = payload
                    state, result = function(state, *args, **kwargs)
                    pipe.send((Message.RESULT, callid, result))
                else:
                    raise KeyError(f"Invalid message: {message}")
        except (EOFError, KeyboardInterrupt):
            return
        except Exception:
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print(f"Error inside process worker: {stacktrace}.", flush=True)
            pipe.send((Message.ERROR, callid, stacktrace))
            return
        finally:
            try:
                pipe.close()
            except Exception:
                pass


class Message(enum.Enum):
    """Enum class representing different message types for parallel communication.

    Enumerated values:
        OK (1): Indicates successful completion or acknowledgment
        RUN (2): Signals to start or execute an operation
        RESULT (3): Represents a result or output message
        STOP (4): Commands to stop or terminate an operation
        ERROR (5): Indicates an error condition or failed operation
    """

    OK = 1
    RUN = 2
    RESULT = 3
    STOP = 4
    ERROR = 5


class Future:
    def __init__(self, receive, callid):
        """Initialize an asynchronous result object.

        Args:
            receive (Callable): The function to receive the result from another process.
            callid (int): A unique identifier for this async call.

        Attributes:
            _receive (Callable): Stored receive function.
            _callid (int): Stored call identifier.
            _result: The result once received, initially None.
            _complete (bool): Flag indicating if result has been received, initially False.
        """
        self._receive = receive
        self._callid = callid
        self._result = None
        self._complete = False

    def __call__(self):
        """
        Retrieves and returns the result of a remote procedure call.

        This method implements the callable interface for handling remote procedure calls.
        If the call hasn't been completed yet, it receives the result using the call ID
        and marks the call as complete. Subsequent calls will return the cached result.

        Returns:
            Any: The result of the remote procedure call

        Note:
            This method is idempotent - multiple calls will return the same result
            after the first call completes.
        """
        if not self._complete:
            self._result = self._receive(self._callid)
            self._complete = True
        return self._result


class Damy:
    """A wrapper class that creates lazy evaluation functions for environment interactions.

    This class wraps an environment and provides lazy evaluation for the step and reset
    methods by returning lambda functions. This can be useful in scenarios where you want
    to delay the actual execution of these operations.

    Args:
        env: The environment to be wrapped. Should implement step() and reset() methods.

    Attributes:
        _env: The wrapped environment instance.

    Example:
        >>> env = SomeEnvironment()
        >>> damy = Damy(env)
        >>> step_fn = damy.step(action)  # Returns a lambda
        >>> result = step_fn()  # Actually executes the step
    """

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        return lambda: self._env.step(action)

    def reset(self):
        return lambda: self._env.reset()
