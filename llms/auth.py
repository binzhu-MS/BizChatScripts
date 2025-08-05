import atexit
import json
import logging
import platform
import os
from msal import PublicClientApplication, SerializableTokenCache

logger = logging.getLogger(__name__)
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class _MsalAuthProvider:
    """
    MSAL-based authentication provider, following RSP's pattern.
    Uses Windows integrated authentication for internal LLM models.
    """
    _SCOPES = None
    _CACHE_FILENAME = None
    INSTANCE = None

    def __init__(self, scopes=None):
        self.scopes = scopes or self.__class__._SCOPES
        assert self.scopes, "No scopes provided for MSAL authentication!"
        
        # Set up token cache
        assert self.__class__._CACHE_FILENAME, "No cache filename provided for MSAL authentication!"
        self.cache = SerializableTokenCache()
        if os.path.exists(self.__class__._CACHE_FILENAME):
            try:
                with open(self.__class__._CACHE_FILENAME, "r", encoding="utf-8") as f:
                    self.cache.deserialize(f.read())
            except (TypeError, FileNotFoundError) as e:
                logger.warning(f"Failed to deserialize cache: {e}. Starting with a fresh cache.")

        atexit.register(lambda: self.serialize_cache() if self.cache.has_state_changed else None)

        # Set up app - using the same configuration as RSP
        self._app = PublicClientApplication(
            client_id='99c1a080-d873-4120-ba44-bd8704143c4a',  # Same as RSP
            authority='https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47',
            enable_broker_on_windows=True,
            enable_broker_on_mac=False,  # Disable on Mac as per RSP
            token_cache=self.cache
        )

        # Force getting the token right away to make sure it gets cached
        self.get_token()

    def serialize_cache(self):
        """Serialize token cache to file."""
        if self.__class__._CACHE_FILENAME:
            try:
                cache_data = self.cache.serialize()
                if not isinstance(cache_data, str):
                    cache_data = json.dumps(cache_data)
                with open(self.__class__._CACHE_FILENAME, "w", encoding="utf-8") as f:
                    f.write(cache_data)
            except TypeError as e:
                logger.error(f"Failed to serialize cache: {e}")

    def get_token(self):
        """Get authentication token using Windows integrated auth."""
        accounts = self._app.get_accounts()
        result = None

        # Try silent authentication first
        if accounts:
            chosen = accounts[0]
            result = self._app.acquire_token_silent(self.scopes, account=chosen)

        # If silent auth fails, use interactive authentication
        if not result:
            if platform.system() == "Linux":
                logger.warning("Linux detected: using device code flow for authentication.")
                flow = self._app.initiate_device_flow(scopes=self.scopes)
                if "user_code" not in flow:
                    raise ValueError("Failed to create device flow. Error: {}".format(json.dumps(flow, indent=4)))
                logger.warning(f"To authenticate, use a web browser to visit {flow['verification_uri']} and enter the code {flow['user_code']}")
                result = self._app.acquire_token_by_device_flow(flow)
            else:
                try:
                    # Windows interactive authentication
                    if self.scopes is not None:
                        result = self._app.acquire_token_interactive(
                            scopes=self.scopes,
                            parent_window_handle=getattr(self._app, "CONSOLE_WINDOW_HANDLE", None)
                        )
                    else:
                        raise ValueError("Scopes are None. Cannot acquire token interactively.")
                except Exception as ex:
                    logger.warning(f"Interactive auth failed: {ex}. Falling back to device code flow.")
                    flow = self._app.initiate_device_flow(scopes=self.scopes)
                    if "user_code" not in flow:
                        raise ValueError("Failed to create device flow. Error: {}".format(json.dumps(flow, indent=4)))
                    logger.warning(f"To authenticate, use a web browser to visit {flow['verification_uri']} and enter the code {flow['user_code']}")
                    result = self._app.acquire_token_by_device_flow(flow)
                    
            if 'error' in result:
                raise ValueError(f"Failed to acquire token. Error: {json.dumps(result, indent=4)}")

        self.serialize_cache()
        return result["access_token"]

    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls.INSTANCE is None:
            cls.INSTANCE = cls()
        return cls.INSTANCE

    @classmethod
    def init(cls):
        """Initialize and return instance."""
        return cls.get_instance()


class LLMAPI(_MsalAuthProvider):
    """LLM API authentication provider using MSAL."""
    _SCOPES = ['https://substrate.office.com/llmapi/LLMAPI.dev']  # Same as RSP
    _CACHE_FILENAME = os.path.join(CURRENT_DIRECTORY, ".msal_token_cache")


# For backward compatibility
class SimpleAuth:
    """Fallback auth for when MSAL is not available."""
    
    def __init__(self, token=None):
        self.token = token or "fallback-token"
    
    def get_token(self):
        return self.token