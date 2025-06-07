"""Type aliases for ROS2 types to improve static typing."""

from typing import Any, Protocol, TypeVar


class _Ros2Message(Protocol):
    """Protocol representing any ROS2 message type."""
    
    def get_fields_and_field_types(self) -> dict:
        """Get the message fields information.
        
        Returns:
            dict: A dictionary of field names and their types
        """
        ...


# TypeVar for any ROS2 message
_Ros2Message_T = TypeVar('_Ros2Message_T', bound=_Ros2Message)


class _Ros2ServiceType(Protocol):
    """Protocol representing a ROS2 service type."""
    
    Request: Any
    Response: Any


# TypeVar for any ROS2 service type
_Ros2ServiceType_T = TypeVar('_Ros2ServiceType_T', bound=_Ros2ServiceType)
