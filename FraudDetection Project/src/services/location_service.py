import requests
import logging

logger = logging.getLogger(__name__)

def get_user_location(ip_address: str) -> dict:
    """
    Fetch geolocation data for the given IP address using ip-api.com.
    Returns a dictionary with city, region, country, and isp.
    Provides a fallback ('Unknown Location') if the API fails or IP is local.
    """
    default_loc = {
        "city": "Unknown",
        "region": "Unknown",
        "country": "Unknown",
        "isp": "Unknown",
        "ip": ip_address
    }
    
    # Return defaults for localhost or unresolvable IPs
    if not ip_address or ip_address in ("127.0.0.1", "::1", "localhost", "0.0.0.0"):
        default_loc["city"] = "Localhost"
        default_loc["isp"] = "Local Network"
        return default_loc
        
    try:
        response = requests.get(f"http://ip-api.com/json/{ip_address}", timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return {
                    "city": data.get("city", "Unknown"),
                    "region": data.get("regionName", "Unknown"),
                    "country": data.get("country", "Unknown"),
                    "isp": data.get("isp", "Unknown"),
                    "ip": ip_address
                }
            else:
                logger.warning(f"IP Geolocation API returned fail status for {ip_address}: {data.get('message')}")
        else:
            logger.warning(f"IP Geolocation API returned status code {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to fetch location for IP {ip_address}: {str(e)}")
        
    return default_loc
