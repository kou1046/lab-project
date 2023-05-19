import os
import sys

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.path.join("api", "db.sqlite3"),
    }
}

INSTALLED_APPS = ("api",)

SECRET_KEY = "REPLACE_ME"

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "orm_setup")
from django.core.wsgi import get_wsgi_application

app = get_wsgi_application()
