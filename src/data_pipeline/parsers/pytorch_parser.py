import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin
import concurrent.futures
from typing import List, Dict, Any, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AdvancedPyTorchParser:
    """
    Deep crawler for PyTorch Sphinx documentation.
    Extracts comprehensive data by navigating from the main API page into individual function/class pages.
    """

    def __init__(self, library_name: str = "pytorch", max_workers: int = 10):
        self.library_name = library_name
        self.max_workers = max_workers
        self.visited_urls = set()
        
        # Standard headers to prevent being blocked by the server
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def get_all_sub_links(self, base_url: str) -> List[str]:
        """
        Scrapes the main API index page to collect all valid links to specific functions/classes.
        """
        logger.info(f"Extracting sub-links from base URL: {base_url}")
        try:
            response = requests.get(base_url, headers=self.headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch base URL: {e}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()

        # Sphinx typical references are within 'a' tags with class 'reference internal'
        for a_tag in soup.find_all('a', class_='reference internal'):
            href = a_tag.get('href')
            if not href or href.startswith('#'):
                continue
            
            # Resolve relative URLs to absolute URLs
            full_url = urljoin(base_url, href)
            
            # Remove URL fragments (e.g., #torch.abs) to avoid scraping the same page multiple times
            clean_url = full_url.split('#')[0]
            
            # Ensure we stay within the pytorch docs domain
            if "pytorch.org/docs" in clean_url:
                links.add(clean_url)

        links_list = list(links)
        logger.info(f"Found {len(links_list)} unique sub-pages to crawl.")
        return links_list

    def parse_detail_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Visits a specific function/class page and extracts deep, comprehensive documentation.
        """
        if url in self.visited_urls:
            return None
        self.visited_urls.add(url)

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Determine the module name from the URL or breadcrumbs
        module_name = "torch"
        nav_tag = soup.find('nav', id='wy-breadcrumbs')
        if nav_tag:
            breadcrumbs = [a.text for a in nav_tag.find_all('a')]
            if len(breadcrumbs) > 1:
                module_name = breadcrumbs[-1]

        # Find the main documentation block (Sphinx wraps details in dl.py class/function)
        main_block = soup.find('dl', class_=re.compile(r'py (function|class|method)'))
        if not main_block:
            return None

        try:
            # 1. Extract Name and Signature
            dt_tag = main_block.find('dt', class_='sig')
            if not dt_tag:
                return None
            
            func_name_tag = dt_tag.find('span', class_='sig-name')
            func_name = func_name_tag.text.strip() if func_name_tag else "Unknown"
            signature = dt_tag.text.strip().replace('¶', '')

            # 2. Navigate to the description body
            dd_tag = main_block.find('dd')
            if not dd_tag:
                return None

            # 3. Extract Main Docstring (Description)
            doc_paragraphs = dd_tag.find_all('p', recursive=False)
            docstring = "\n".join([p.text.strip() for p in doc_paragraphs])

            # 4. Extract Parameters and Returns deeply
            parameters = {}
            returns = ""
            field_lists = dd_tag.find_all('dl', class_='field-list')
            
            for field_list in field_lists:
                dts = field_list.find_all('dt')
                dds = field_list.find_all('dd')
                
                for dt, dd in zip(dts, dds):
                    field_name = dt.text.strip().replace(':', '')
                    field_desc = dd.text.strip()
                    
                    if "Return" in field_name or "Rtype" in field_name:
                        returns += f"{field_desc}\n"
                    else:
                        parameters[field_name] = field_desc

            # 5. Extract Code Examples (Crucial for Fine-Tuning)
            examples = ""
            example_blocks = dd_tag.find_all('div', class_='doctest highlight-default notranslate')
            for block in example_blocks:
                examples += block.text.strip() + "\n\n"

            # 6. Build Rich Search Text (Context for RAG)
            search_text = (
                f"Function: {func_name}\n"
                f"Signature: {signature}\n"
                f"Description: {docstring}\n"
                f"Parameters: {json.dumps(parameters, ensure_ascii=False)}\n"
                f"Returns: {returns}\n"
                f"Examples:\n{examples}"
            )

            return {
                "library_name": self.library_name,
                "module_name": module_name,
                "func_name": func_name,
                "signature": signature,
                "docstring": docstring,
                "parameters": parameters,
                "returns": returns.strip(),
                "examples": examples.strip(),
                "source_url": url,
                "search_text": search_text
            }

        except Exception as e:
            logger.warning(f"Error parsing detail page {url}: {e}")
            return None

    def crawl_and_extract(self, base_url: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Orchestrates the crawling process using multithreading.
        """
        sub_links = self.get_all_sub_links(base_url)
        
        if limit:
            logger.info(f"Applying limit: Scraping only {limit} pages for testing.")
            sub_links = sub_links[:limit]

        results = []
        logger.info(f"Starting deep extraction on {len(sub_links)} pages using {self.max_workers} threads...")

        # Use ThreadPoolExecutor to scrape multiple pages concurrently (Fast!)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Map the parse_detail_page function to all sub_links
            future_to_url = {executor.submit(self.parse_detail_page, url): url for url in sub_links}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_url), 1):
                url = future_to_url[future]
                try:
                    data = future.result()
                    if data:
                        results.append(data)
                        if i % 50 == 0:
                            logger.info(f"Progress: Processed {i}/{len(sub_links)} pages.")
                except Exception as e:
                    logger.error(f"Thread error on {url}: {e}")

        logger.info(f"Crawling complete. Successfully extracted {len(results)} valid records.")
        return results

    def save_to_json(self, data: List[Dict[str, Any]], output_path: str):
        """Saves the extracted records to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved {len(data)} records to {output_path}")

# --- Test the Deep Crawler ---
if __name__ == "__main__":
    parser = AdvancedPyTorchParser(max_workers=10)
    
    # URL gốc chứa danh sách các hàm
    base_url = "https://docs.pytorch.org/docs/2.12/pytorch-api.html" 
    
    # Lưu ý: Set limit=50 khi chạy test để không phải đợi lâu. 
    # Bỏ limit=None để cào TOÀN BỘ documentation.
    data = parser.crawl_and_extract(base_url) 
    
    parser.save_to_json(data, "data/raw/pytorch_deep_raw.json")