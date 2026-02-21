# logger module v1.3
from scripts.time_log import time_log_module as tlm, time_log_module_files as tlmf
import time, os
import requests
from urllib.parse import urlparse

def upload2web(path: str) -> str:
    with open(path, "rb") as f:
        r = requests.post("https://0x0.st", files={"file": f})
    if r.status_code == 200:
        return str(r.text.strip())  # lien direct
    else:
        return ""


def webhook_post(content, webhook_url: str, username: str="", image_url: str=""):
    data = {"content": content}
    if not username == "" and isinstance(username, str):
        data["username"] = username
    if not image_url == "" and isinstance(image_url, str):
        data["avatar_url"] = image_url
    response = requests.post(webhook_url, json=data)
    return response
    
class logger():
    def __init__(self, discord_webhook: str="", name: str="", icon: str=""):
        if not os.path.isdir("logs"): # create logs folder if not exists
            os.makedirs("logs")
        self.start = str(time.time())
        self.actual_log = f"{tlmf()}.log"
        self.discord_webhook = discord_webhook
        self.name = name

        if icon == "":
            self.image_url = ""
        else:
            parsed = urlparse(icon)
            if parsed.scheme in ("http", "https") and parsed.netloc:
                self.image_url = icon
            else:
                self.image_url = upload2web(icon)
        
        try:
            with open(rf"logs\{self.actual_log}", "w") as file:
                file.write(f"Start point [{self.start}]\n")
        except FileNotFoundError:
            with open(rf"logs\{self.actual_log}", "x") as file:
                file.write(f"Start point [{self.start}]\n")
        
    def log(self, data: str, v: bool=True, Wh: bool=True, mention: bool=False):
        if not isinstance(data, str):
            with open(rf"logs\{self.actual_log}", "a") as file:
                file.write(f"{tlm()} The input data for the log function is not a string. The data will NOT be logged.\n")
            raise Warning(f"{tlm()} The input data for the log function is not a string. The data will NOT be logged.")
        
        with open(rf"logs\{self.actual_log}", "a") as file:
            file.write(f"{tlm()} {data}\n")

        if v:
            print(f"{tlm()} {data}")
        
        if Wh and not self.discord_webhook == "":
            if mention:
                tmp = webhook_post(f"{tlm()} - ||@everyone|| {data}", self.discord_webhook, username=self.name, image_url=self.image_url)
            else:
                tmp = webhook_post(f"{tlm()} {data}", self.discord_webhook, username=self.name, image_url=self.image_url)

    