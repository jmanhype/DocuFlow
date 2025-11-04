"""
Message Queue Implementation for DocuFlow Service Colony

This module provides message queue functionality for inter-service communication
using Dapr pub/sub patterns.
"""

import json
import logging
from typing import Dict, Any, Callable, Optional
from dapr.clients import DaprClient

logger = logging.getLogger(__name__)


class MessageQueue:
    """
    Message queue wrapper for Dapr pub/sub functionality.

    This class provides a simplified interface for publishing and subscribing
    to messages across the service colony.
    """

    def __init__(self, pubsub_name: str = "pubsub"):
        """
        Initialize the message queue.

        Args:
            pubsub_name: The name of the Dapr pub/sub component (default: "pubsub")
        """
        self.pubsub_name = pubsub_name
        self.client: Optional[DaprClient] = None

    def __enter__(self):
        """Context manager entry."""
        self.client = DaprClient()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.client:
            self.client.close()

    def publish(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Publish a message to a topic.

        Args:
            topic: The topic name to publish to
            data: The message data to publish

        Raises:
            Exception: If publishing fails
        """
        try:
            if not self.client:
                raise RuntimeError("MessageQueue not initialized. Use as context manager.")

            logger.info(f"Publishing message to topic '{topic}'")
            self.client.publish_event(
                pubsub_name=self.pubsub_name,
                topic_name=topic,
                data=json.dumps(data)
            )
            logger.debug(f"Message published successfully to '{topic}'")

        except Exception as e:
            logger.error(f"Failed to publish message to topic '{topic}': {e}")
            raise

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to a topic with a handler function.

        Note: This is a placeholder for the subscription pattern.
        In practice, subscriptions are handled via Dapr's FastAPI extension
        using the @dapr_app.subscribe decorator.

        Args:
            topic: The topic name to subscribe to
            handler: The callback function to handle messages
        """
        logger.info(f"Subscription requested for topic '{topic}'")
        logger.warning(
            "Direct subscription not supported. Use @dapr_app.subscribe decorator "
            "in your FastAPI application instead."
        )


def publish_event(pubsub_name: str, topic: str, data: Dict[str, Any]) -> None:
    """
    Convenience function to publish a single event.

    Args:
        pubsub_name: The name of the pub/sub component
        topic: The topic name
        data: The event data
    """
    with MessageQueue(pubsub_name) as mq:
        mq.publish(topic, data)
