import logging
import pystache

logger = logging.getLogger(__name__)


def render_messages(input_data, variables):
    """
    Render message templates with variables using Mustache templating.
    
    Args:
        input_data: Messages dict or list of messages
        variables: Dictionary of variables to substitute
        
    Returns:
        Formatted messages dictionary
    """
    if isinstance(input_data, dict):
        messages = input_data['messages']
        result = dict(input_data)
    else:
        messages = input_data
        result = {'messages': messages}
    
    result['messages'] = []
    for message in messages:
        formatted_message = dict(message)
        if 'content' in formatted_message:
            formatted_message['content'] = pystache.render(message['content'], variables)
        result['messages'].append(formatted_message)
    
    return result