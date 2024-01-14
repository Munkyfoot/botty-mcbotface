"""A wrapper for the Wikipedia API that allows for searching and requesting pages."""

import json

import requests
import wikipediaapi
from dotenv import load_dotenv

load_dotenv()


class WikiAPI:
    def __init__(self):
        self.base_url = "https://api.wikimedia.org/core/v1/wikipedia/"
        self.view_base_url = "https://en.wikipedia.org/wiki/"
        self.lang = "en"
        self.wiki = wikipediaapi.Wikipedia(self.lang)

    def search(self, query: str, limit: int = 10) -> dict:
        """Search for a page."""
        url = self.base_url + f"{self.lang}/search/page"
        params = {"q": query, "limit": limit}

        try:
            response = requests.get(url, params=params)
        except Exception as e:
            print(e)
            return None

        if response.status_code == 200:
            json_load = json.loads(response.text)
            return json_load
        else:
            print(response.status_code)
            return None

    def request_page(self, key: str) -> dict:
        """Get a page by its key."""
        url = self.base_url + f"{self.lang}/page/{key}/bare"
        params = {"content_model": "json"}

        try:
            response = requests.get(url, params=params)
        except Exception as e:
            print(e)
            return None

        if response.status_code == 200:
            json_load = json.loads(response.text)
            return json_load
        else:
            print(response.status_code)
            return None

    def request_page_html(self, key: str) -> str:
        page = self.request_page(key)

        if page is not None:
            response = requests.get(page["html_url"])
            if response.status_code == 200:
                print(response.text)
            else:
                print(response.status_code)
                return None
        else:
            print("Page not found.")
            return None

    def request_page_source(self, key: str) -> dict:
        """Get a page by its key."""
        url = self.base_url + f"{self.lang}/page/{key}"
        params = {"content_model": "json"}

        try:
            response = requests.get(url, params=params)
        except Exception as e:
            print(e)
            return None

        if response.status_code == 200:
            json_load = json.loads(response.text).get("source", None)
            return json_load
        else:
            print(response.status_code)
            return None

    def get_page(self, key: str) -> wikipediaapi.WikipediaPage:
        page = self.wiki.page(key)
        if page.exists():
            return page
        else:
            print("Page not found.")
            return None

    def get_view_url(self, key: str) -> str:
        return self.view_base_url + key
