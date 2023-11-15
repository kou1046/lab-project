import os
from dotenv import load_dotenv

load_dotenv("./.env")

""" 
django の DB, ORM のみを使いたい場合, このファイルをインポートする．
"""

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": os.environ["MYSQL_DATABASE"],
        "USER": os.environ["MYSQL_USER"],
        "PASSWORD": os.environ["MYSQL_PASSWORD"],
        "HOST": "localhost",
        "PORT": "3306",
    },
    "2023": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": os.environ["MYSQL_DATABASE"],
        "USER": os.environ["MYSQL_USER"],
        "PASSWORD": os.environ["MYSQL_PASSWORD"],
        "HOST": "localhost",
        "PORT": "3307",
    },
}

INSTALLED_APPS = ("api",)

SECRET_KEY = "REPLACE_ME"

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "db_setup")
from django.core.wsgi import get_wsgi_application

app = get_wsgi_application()
