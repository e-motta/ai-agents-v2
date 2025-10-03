import time
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from llama_index.core import Document

from app.core.logging import get_logger
from app.core.settings import get_settings

logger = get_logger(__name__)

_settings = get_settings()
BASE_URL = _settings.BASE_URL
REQUEST_HEADERS = _settings.REQUEST_HEADERS


def _scrape_page_content(url: str) -> dict[str, Any]:
    """
    Scrape content from a single page.

    Args:
        url: The URL to scrape

    Returns:
        Dictionary containing text content and metadata
    """
    try:
        logger.info("Scraping content from URL", url=url)

        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract text content
        text = soup.get_text()

        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = " ".join(chunk for chunk in chunks if chunk)

        logger.info(
            "Successfully scraped content", url=url, content_length=len(cleaned_text)
        )

        return {"content": cleaned_text, "url": url}
    except Exception as e:
        logger.exception("Error scraping URL", url=url, error=str(e))
        raise


def _find_collection_links(base_url: str) -> set[str]:
    """
    Find all collection links from the main help center page.

    Args:
        base_url: The main help center URL

    Returns:
        Set of collection URLs
    """
    collection_links = set()

    try:
        logger.info("Finding collection links", base_url=base_url)

        response = requests.get(base_url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Find all links
        links = soup.find_all("a", href=True)

        for link in links:
            href = str(link["href"])
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)

            # Check if this is a collection link
            if "/collections/" in absolute_url:
                collection_links.add(absolute_url)
                logger.info("Found collection link", url=absolute_url)

        logger.info(
            "Collection links search completed",
            base_url=base_url,
            links_found=len(collection_links),
        )
        return collection_links
    except Exception as e:
        logger.exception(
            "Error finding collection links", base_url=base_url, error=str(e)
        )
        return set()


def _find_article_links(collection_url: str) -> set[str]:
    """
    Find all article links from a collection page.

    Args:
        collection_url: The collection URL to crawl

    Returns:
        Set of article URLs
    """
    article_links = set()

    try:
        logger.info("Finding article links", collection_url=collection_url)

        response = requests.get(collection_url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Find all links
        links = soup.find_all("a", href=True)

        for link in links:
            href = str(link["href"])
            # Convert relative URLs to absolute
            absolute_url = urljoin(collection_url, href)

            # Check if this is an article link
            if "/articles/" in absolute_url:
                article_links.add(absolute_url)
                logger.info("Found article link", url=absolute_url)

        logger.info(
            "Article links search completed",
            collection_url=collection_url,
            links_found=len(article_links),
        )
        return article_links
    except Exception as e:
        logger.exception(
            "Error finding article links", collection_url=collection_url, error=str(e)
        )
        return set()


def crawl_help_center() -> list[Document]:
    """
    Perform comprehensive crawling of the InfinitePay help center.

    Returns:
        List of LlamaIndex Document objects
    """
    documents = []
    visited_urls = set()

    try:
        start_time = time.time()
        logger.info("Starting comprehensive crawl of InfinitePay help center")

        # Step 1: Find all collection links
        collection_links = _find_collection_links(BASE_URL)

        # Step 2: Find all article links from collections
        all_article_links = set()
        for collection_url in collection_links:
            article_links = _find_article_links(collection_url)
            all_article_links.update(article_links)

        logger.info(
            "Total unique article links found", total_links=len(all_article_links)
        )

        # Step 3: Process each article
        for i, article_url in enumerate(all_article_links, 1):
            if article_url in visited_urls:
                continue

            visited_urls.add(article_url)

            try:
                logger.info(
                    "Processing article",
                    current=i,
                    total=len(all_article_links),
                    url=article_url,
                )

                # Scrape the article content
                page_data = _scrape_page_content(article_url)

                if page_data["content"].strip():
                    # Create LlamaIndex Document
                    doc = Document(
                        text=page_data["content"],
                        metadata={
                            "url": article_url,
                            "source": "infinitepay_help_center",
                        },
                    )
                    documents.append(doc)
                    logger.info("Created document", url=article_url)
                else:
                    logger.warning("No content found", url=article_url)

            except Exception as e:
                logger.exception(
                    "Error processing article", url=article_url, error=str(e)
                )
                continue

        execution_time = time.time() - start_time
        logger.info(
            "Crawling completed",
            documents_created=len(documents),
            execution_time=execution_time,
        )
        return documents
    except Exception as e:
        logger.exception("Error during crawling", error=str(e))
        raise
