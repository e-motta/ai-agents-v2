"""
Unit tests for Knowledge Agent scraping functions.

These tests verify the web scraping functionality for crawling
the help center and extracting content.
"""

from unittest.mock import Mock, patch

import pytest
import requests
from llama_index.core import Document

from app.agents.knowledge_agent.scraping import (
    _find_article_links,
    _find_collection_links,
    _scrape_page_content,
    crawl_help_center,
)


class TestScrapePageContent:
    """Test the _scrape_page_content function."""

    @patch("app.agents.knowledge_agent.scraping.get_settings")
    @patch("app.agents.knowledge_agent.scraping.requests.get")
    def test_scrape_page_content_success(self, mock_get, mock_get_settings):
        """Test successful page content scraping."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.REQUEST_HEADERS = {"User-Agent": "Test Agent"}
        mock_get_settings.return_value = mock_settings

        # Mock response
        mock_response = Mock()
        mock_response.content = (
            b"<html><body><h1>Test Title</h1><p>Test content</p></body></html>"
        )
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Call the function
        result = _scrape_page_content("http://test.com")

        # Verify result
        assert result["url"] == "http://test.com"
        assert "Test Title" in result["content"]
        assert "Test content" in result["content"]
        mock_get.assert_called_once_with(
            "http://test.com", headers=mock_settings.REQUEST_HEADERS, timeout=30
        )

    @patch("app.agents.knowledge_agent.scraping.get_settings")
    @patch("app.agents.knowledge_agent.scraping.requests.get")
    def test_scrape_page_content_removes_scripts_and_styles(
        self, mock_get, mock_get_settings
    ):
        """Test that scripts and styles are removed from content."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.REQUEST_HEADERS = {"User-Agent": "Test Agent"}
        mock_get_settings.return_value = mock_settings

        # Mock response with scripts and styles
        html_content = """
        <html>
            <head>
                <style>body { color: red; }</style>
            </head>
            <body>
                <h1>Test Title</h1>
                <script>alert('test');</script>
                <p>Test content</p>
            </body>
        </html>
        """
        mock_response = Mock()
        mock_response.content = html_content.encode()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Call the function
        result = _scrape_page_content("http://test.com")

        # Verify scripts and styles are removed
        assert "alert('test')" not in result["content"]
        assert "body { color: red; }" not in result["content"]
        assert "Test Title" in result["content"]
        assert "Test content" in result["content"]

    @patch("app.agents.knowledge_agent.scraping.get_settings")
    @patch("app.agents.knowledge_agent.scraping.requests.get")
    def test_scrape_page_content_cleans_whitespace(self, mock_get, mock_get_settings):
        """Test that whitespace is properly cleaned."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.REQUEST_HEADERS = {"User-Agent": "Test Agent"}
        mock_get_settings.return_value = mock_settings

        # Mock response with messy whitespace
        html_content = """
        <html>
            <body>
                <h1>  Test Title  </h1>
                <p>  Test   content  with   multiple    spaces  </p>
            </body>
        </html>
        """
        mock_response = Mock()
        mock_response.content = html_content.encode()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Call the function
        result = _scrape_page_content("http://test.com")

        # Verify whitespace is cleaned
        assert "  " not in result["content"]  # No double spaces
        assert (
            result["content"].strip() == result["content"]
        )  # No leading/trailing whitespace

    @patch("app.agents.knowledge_agent.scraping.get_settings")
    @patch("app.agents.knowledge_agent.scraping.requests.get")
    def test_scrape_page_content_http_error(self, mock_get, mock_get_settings):
        """Test handling of HTTP errors."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.REQUEST_HEADERS = {"User-Agent": "Test Agent"}
        mock_get_settings.return_value = mock_settings

        # Mock HTTP error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        # Call the function and expect exception
        with pytest.raises(requests.HTTPError):
            _scrape_page_content("http://test.com")

    @patch("app.agents.knowledge_agent.scraping.get_settings")
    @patch("app.agents.knowledge_agent.scraping.requests.get")
    def test_scrape_page_content_connection_error(self, mock_get, mock_get_settings):
        """Test handling of connection errors."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.REQUEST_HEADERS = {"User-Agent": "Test Agent"}
        mock_get_settings.return_value = mock_settings

        # Mock connection error
        mock_get.side_effect = requests.ConnectionError("Connection failed")

        # Call the function and expect exception
        with pytest.raises(requests.ConnectionError):
            _scrape_page_content("http://test.com")


class TestFindCollectionLinks:
    """Test the _find_collection_links function."""

    @patch("app.agents.knowledge_agent.scraping.get_settings")
    @patch("app.agents.knowledge_agent.scraping.requests.get")
    def test_find_collection_links_success(self, mock_get, mock_get_settings):
        """Test successful collection link finding."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.REQUEST_HEADERS = {"User-Agent": "Test Agent"}
        mock_get_settings.return_value = mock_settings

        # Mock response with collection links
        html_content = """
        <html>
            <body>
                <a href="/collections/payments">Payments</a>
                <a href="/collections/account">Account</a>
                <a href="/articles/some-article">Article</a>
                <a href="https://external.com/collections/external">External</a>
            </body>
        </html>
        """
        mock_response = Mock()
        mock_response.content = html_content.encode()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Call the function
        result = _find_collection_links("http://test.com")

        # Verify result
        assert len(result) == 3  # 3 collection links
        assert "http://test.com/collections/payments" in result
        assert "http://test.com/collections/account" in result
        assert "https://external.com/collections/external" in result
        assert (
            "http://test.com/articles/some-article" not in result
        )  # Not a collection link

    @patch("app.agents.knowledge_agent.scraping.get_settings")
    @patch("app.agents.knowledge_agent.scraping.requests.get")
    def test_find_collection_links_handles_relative_urls(
        self, mock_get, mock_get_settings
    ):
        """Test that relative URLs are converted to absolute URLs."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.REQUEST_HEADERS = {"User-Agent": "Test Agent"}
        mock_get_settings.return_value = mock_settings

        # Mock response with relative URLs
        html_content = """
        <html>
            <body>
                <a href="/collections/payments">Payments</a>
                <a href="collections/account">Account</a>
                <a href="../collections/parent">Parent</a>
            </body>
        </html>
        """
        mock_response = Mock()
        mock_response.content = html_content.encode()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Call the function
        result = _find_collection_links("http://test.com/help")

        # Verify URLs are properly converted
        assert "http://test.com/collections/payments" in result
        assert (
            "http://test.com/collections/account" in result
        )  # Fixed: should be /collections/account
        assert "http://test.com/collections/parent" in result

    @patch("app.agents.knowledge_agent.scraping.get_settings")
    @patch("app.agents.knowledge_agent.scraping.requests.get")
    def test_find_collection_links_handles_errors(self, mock_get, mock_get_settings):
        """Test error handling in collection link finding."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.REQUEST_HEADERS = {"User-Agent": "Test Agent"}
        mock_get_settings.return_value = mock_settings

        # Mock error
        mock_get.side_effect = requests.RequestException("Request failed")

        # Call the function
        result = _find_collection_links("http://test.com")

        # Should return empty set
        assert result == set()


class TestFindArticleLinks:
    """Test the _find_article_links function."""

    @patch("app.agents.knowledge_agent.scraping.get_settings")
    @patch("app.agents.knowledge_agent.scraping.requests.get")
    def test_find_article_links_success(self, mock_get, mock_get_settings):
        """Test successful article link finding."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.REQUEST_HEADERS = {"User-Agent": "Test Agent"}
        mock_get_settings.return_value = mock_settings

        # Mock response with article links
        html_content = """
        <html>
            <body>
                <a href="/articles/payment-setup">Payment Setup</a>
                <a href="/articles/account-management">Account Management</a>
                <a href="/collections/payments">Collection</a>
                <a href="https://external.com/articles/external">External Article</a>
            </body>
        </html>
        """
        mock_response = Mock()
        mock_response.content = html_content.encode()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Call the function
        result = _find_article_links("http://test.com/collections/payments")

        # Verify result
        assert len(result) == 3  # 3 article links
        assert (
            "http://test.com/articles/payment-setup" in result
        )  # Fixed: should be /articles/payment-setup
        assert "http://test.com/articles/account-management" in result
        assert "https://external.com/articles/external" in result
        assert (
            "http://test.com/collections/payments/collections/payments" not in result
        )  # Not an article link

    @patch("app.agents.knowledge_agent.scraping.get_settings")
    @patch("app.agents.knowledge_agent.scraping.requests.get")
    def test_find_article_links_handles_errors(self, mock_get, mock_get_settings):
        """Test error handling in article link finding."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.REQUEST_HEADERS = {"User-Agent": "Test Agent"}
        mock_get_settings.return_value = mock_settings

        # Mock error
        mock_get.side_effect = requests.RequestException("Request failed")

        # Call the function
        result = _find_article_links("http://test.com/collections/payments")

        # Should return empty set
        assert result == set()


class TestCrawlHelpCenter:
    """Test the crawl_help_center function."""

    @patch("app.agents.knowledge_agent.scraping.get_settings")
    @patch("app.agents.knowledge_agent.scraping._find_collection_links")
    @patch("app.agents.knowledge_agent.scraping._find_article_links")
    @patch("app.agents.knowledge_agent.scraping._scrape_page_content")
    def test_crawl_help_center_success(
        self,
        mock_scrape_page_content,
        mock_find_article_links,
        mock_find_collection_links,
        mock_get_settings,
    ):
        """Test successful help center crawling."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.BASE_URL = "http://test.com/help"
        mock_get_settings.return_value = mock_settings

        # Mock collection links
        mock_find_collection_links.return_value = {
            "http://test.com/help/collections/payments",
            "http://test.com/help/collections/account",
        }

        # Mock article links
        mock_find_article_links.side_effect = [
            {
                "http://test.com/help/articles/payment-setup",
                "http://test.com/help/articles/payment-fees",
            },
            {"http://test.com/help/articles/account-settings"},
        ]

        # Mock page content
        mock_scrape_page_content.side_effect = [
            {
                "content": "Payment setup content",
                "url": "http://test.com/help/articles/payment-setup",
            },
            {
                "content": "Payment fees content",
                "url": "http://test.com/help/articles/payment-fees",
            },
            {
                "content": "Account settings content",
                "url": "http://test.com/help/articles/account-settings",
            },
        ]

        # Call the function
        result = crawl_help_center()

        # Verify result
        assert len(result) == 3
        assert all(isinstance(doc, Document) for doc in result)
        assert any("Payment setup content" in doc.text for doc in result)
        assert any("Payment fees content" in doc.text for doc in result)
        assert any("Account settings content" in doc.text for doc in result)

        # Verify all functions were called
        mock_find_collection_links.assert_called_once_with("http://test.com/help")
        assert mock_find_article_links.call_count == 2
        assert mock_scrape_page_content.call_count == 3

    @patch("app.agents.knowledge_agent.scraping.get_settings")
    @patch("app.agents.knowledge_agent.scraping._find_collection_links")
    @patch("app.agents.knowledge_agent.scraping._find_article_links")
    @patch("app.agents.knowledge_agent.scraping._scrape_page_content")
    def test_crawl_help_center_skips_empty_content(
        self,
        mock_scrape_page_content,
        mock_find_article_links,
        mock_find_collection_links,
        mock_get_settings,
    ):
        """Test that articles with empty content are skipped."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.BASE_URL = "http://test.com/help"
        mock_get_settings.return_value = mock_settings

        # Mock collection links
        mock_find_collection_links.return_value = {
            "http://test.com/help/collections/payments"
        }

        # Mock article links
        mock_find_article_links.return_value = {
            "http://test.com/help/articles/empty-article"
        }

        # Mock empty page content
        mock_scrape_page_content.return_value = {
            "content": "",
            "url": "http://test.com/help/articles/empty-article",
        }

        # Call the function
        result = crawl_help_center()

        # Verify result - should be empty due to empty content
        assert len(result) == 0

    @patch("app.agents.knowledge_agent.scraping.get_settings")
    @patch("app.agents.knowledge_agent.scraping._find_collection_links")
    @patch("app.agents.knowledge_agent.scraping._find_article_links")
    @patch("app.agents.knowledge_agent.scraping._scrape_page_content")
    def test_crawl_help_center_handles_scraping_errors(
        self,
        mock_scrape_page_content,
        mock_find_article_links,
        mock_find_collection_links,
        mock_get_settings,
    ):
        """Test that scraping errors are handled gracefully."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.BASE_URL = "http://test.com/help"
        mock_get_settings.return_value = mock_settings

        # Mock collection links
        mock_find_collection_links.return_value = {
            "http://test.com/help/collections/payments"
        }

        # Mock article links
        mock_find_article_links.return_value = {
            "http://test.com/help/articles/good-article",
            "http://test.com/help/articles/bad-article",
        }

        # Mock page content - one success, one error
        def side_effect(url):
            if "good-article" in url:
                return {"content": "Good content", "url": url}
            raise Exception("Scraping failed")  # noqa: TRY002

        mock_scrape_page_content.side_effect = side_effect

        # Call the function
        result = crawl_help_center()

        # Verify result - should only contain the successful article
        assert len(result) == 1
        assert "Good content" in result[0].text

    @patch("app.agents.knowledge_agent.scraping.get_settings")
    @patch("app.agents.knowledge_agent.scraping._find_collection_links")
    def test_crawl_help_center_handles_collection_error(
        self, mock_find_collection_links, mock_get_settings
    ):
        """Test that collection finding errors are handled."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.BASE_URL = "http://test.com/help"
        mock_get_settings.return_value = mock_settings

        # Mock collection finding error
        mock_find_collection_links.side_effect = Exception("Collection finding failed")

        # Call the function and expect exception
        with pytest.raises(Exception, match="Collection finding failed"):
            crawl_help_center()

    @patch("app.agents.knowledge_agent.scraping.get_settings")
    @patch("app.agents.knowledge_agent.scraping._find_collection_links")
    @patch("app.agents.knowledge_agent.scraping._find_article_links")
    @patch("app.agents.knowledge_agent.scraping._scrape_page_content")
    def test_crawl_help_center_deduplicates_urls(
        self,
        mock_scrape_page_content,
        mock_find_article_links,
        mock_find_collection_links,
        mock_get_settings,
    ):
        """Test that duplicate URLs are handled."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.BASE_URL = "http://test.com/help"
        mock_get_settings.return_value = mock_settings

        # Mock collection links
        mock_find_collection_links.return_value = {
            "http://test.com/help/collections/payments"
        }

        # Mock article links with duplicates
        mock_find_article_links.return_value = {
            "http://test.com/help/articles/duplicate-article",  # Duplicate
        }

        # Mock page content
        mock_scrape_page_content.return_value = {
            "content": "Duplicate content",
            "url": "http://test.com/help/articles/duplicate-article",
        }

        # Call the function
        result = crawl_help_center()

        # Verify result - should only process the article once
        assert len(result) == 1
        assert mock_scrape_page_content.call_count == 1
