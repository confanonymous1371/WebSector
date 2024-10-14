from collections import defaultdict
from urllib.parse import urlparse

import datetime
import scrapy
import csv
import json
from scrapy import signals  # Importing signals
import tempfile
import shutil
import os
import re
from urllib.parse import urlparse, urlunparse

import csv
from twisted.internet.error import TimeoutError, DNSLookupError
from scrapy.spidermiddlewares.httperror import HttpError

class ModifiedPrivacyPolicySpider(scrapy.Spider):
    name = "linkedin_spider"
    scraped_items = defaultdict(dict)  # Dictionary to store all scraped items
    current_domain = None  # Variable to keep track of the current domain
    #change
    # run 4 again
    #2,4,6
    # 66 is filleddddd
    file_number = 98

    count = 0


    def __init__(self, *args, **kwargs):
        super(ModifiedPrivacyPolicySpider, self).__init__(*args, **kwargs)
        self.start_urls_with_sectors = defaultdict(dict)
        self.start_urls_with_sectors_norm = defaultdict(dict)
        self.read_urls_from_csv()
        self.read_urls_from_csv1()

    def read_urls_from_csv(self):
        # Replace 'path_to_your_csv' with the actual path to your CSV file
        #change
        #2,4,6
        with open(f'/home/sxs7285/data/linkedin_prj/Initial_Linkedin_for_Scarping/Chunked_labeled_data/non_nan_entries_part_{self.file_number}.csv', 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader) 
            print("0")

            for row in reader:
                
                self.start_urls_with_sectors[row[0]] = row[1]
                self.start_urls_with_sectors_norm[normalize_url(row[0])] = row[1]

        print("00")


    print("01")

    def read_urls_from_csv1(self):
        with open(f'/home/sxs7285/data/linkedin_prj/Initial_Linkedin_for_Scarping/Chunked_labeled_data/non_nan_entries_part_{self.file_number}.csv', 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip the header row
            self.start_urls =[row[0] for row in reader]
       
    
    print("02")

    # print("start_urls",len(self.start_urls))

    def parse(self, response):
        # print("1")

        if len(self.start_urls_with_sectors[response.url]) !=0:
          extracted_sector = self.start_urls_with_sectors[response.url]
        else:
          extracted_sector = self.start_urls_with_sectors_norm[normalize_url(response.url)]

        # print("1")
        #print("response.url ------",response.url)
        print("response.url ------",self.count,response.url)
        self.count = self.count + 1


        if type(extracted_sector) != dict:
            yield scrapy.Request(response.url, callback=self.parse_main_page1, errback=self.handle_error,
            meta={'sector_of_activity': extracted_sector})

        else: 
            print("**empty label", response.url)

  
   
    def parse_main_page1(self, response):
      sector_of_activity = response.meta.get('sector_of_activity', None)  # Getting additional data from the meta
      pp_url = response.url
      main_page_url = response.url.rsplit('/', 2)[0]
      
      keywords = ['privacy', 'policy', 'term', 'condition']  # Keywords to look for in the URLs or anchor texts
      filtered_links = []
    #   print("2")

      all_links = response.xpath('//a/@href').extract()
      
      # Get the domain of the current page
      self.current_domain = urlparse(response.url).netloc
      
      for url in all_links:
          # Resolve relative links to absolute URLs
          absolute_url = response.urljoin(url)
          link_domain = urlparse(absolute_url).netloc  # Get the domain of the hyperlink
          
          # print("link_domain == current_domain",link_domain != current_domain)
          if link_domain != self.current_domain:
              # This hyperlink points to the same website
              pass  # or whatever action you want to take
          else:
              # This hyperlink points to a different website
              # Filtering links based on keywords
              if not any(re.search(keyword, url, re.IGNORECASE) for keyword in keywords):
                  filtered_links.append(absolute_url)
      
      # Add the main page URL to the list of links
      all_links = filtered_links + [response.url]
      all_links = list(set(all_links)) 
      # Create tuples of links and the main page URL
      all_links_tuple = [(link, pp_url) for link in all_links]
      
      for link_tuple in all_links_tuple:
          link, pp_link = link_tuple
          yield scrapy.Request(link, callback=self.parse_linked_page, errback=self.handle_error, meta={'pp_url': pp_link, 'sector_of_activity':sector_of_activity})


    

    def parse_linked_page(self, response):

        new_domain = urlparse(response.url).netloc
        
        pp_url = response.meta.get('pp_url', None)  # Getting additional data from the meta
        sector_of_activity = response.meta.get('sector_of_activity', None)  # Getting additional data from the meta

        pp_domain = urlparse(pp_url).netloc if pp_url else None  # Extract domain from pp_url
        if self.current_domain and self.current_domain != pp_domain:
            self.save_to_file(self.scraped_items, self.current_domain.replace('.', '_'))
            self.scraped_items.clear()  # Clear the current data
        else:
          pass

               
        self.current_domain = pp_domain 

        item = self.extract_content_with_title(response)
        main_page_url = response.url.rsplit('/', 2)[0]
        if not item or 'content' not in item:
            return

        parsed_url = pp_url
        main_page_dir = self.current_domain.replace('.', '_')

        main_sanitized_url = str(main_page_dir)



        sanitized_url = response.url.replace('http://', '').replace('https://', '').replace('/', '_').replace('.', '_')
        self.scraped_items[main_sanitized_url][sanitized_url] = {
            "url": response.url,
            "date_scraped": datetime.datetime.now().isoformat(),
            "title": item['title'],
            "content": item['content'],
            "pp_url": pp_url,
            "sector_of_activity":sector_of_activity
        }


        file_name = extract_filename(response.url)
        file_path = f'{file_name}.html'
        self.save_to_html(response, file_path,main_page_url)
    

    def save_to_file(self, website_data, file_name):
      
      #change
      file_path = f'/home/sxs7285/data/linkedin_prj/Initial_Linkedin_for_Scarping/Scrapped_data/jsons/json_{str(self.file_number)}/{file_name}.json'
      # Read the existing data or initialize an empty dictionary if the file doesn't exist
      try:
          with open(file_path, 'r', encoding='utf-8') as f:
              existing_data = json.load(f)
      except (FileNotFoundError, json.JSONDecodeError):
          existing_data = {}

      # Update the existing data with the new data
      for main_key, nested_data in website_data.items():
          if main_key in existing_data:
              existing_data[main_key].update(nested_data)
          else:
              existing_data[main_key] = nested_data

      # Write the updated data back to the file
      with open(file_path, 'w', encoding='utf-8') as f:
          json.dump(existing_data, f, ensure_ascii=False, indent=4)


    def spider_closed(self, spider):
        print("BYEEEEEE00")

        if self.scraped_items:  # Check if there's any remaining data
            print("BYEEEEEE")
            self.save_to_file(self.scraped_items, self.current_domain.replace('.', '_'))

  

    def extract_content_with_title(self, response):

        # Extract title and entire content from the page
        title = response.xpath('//title/text()').extract_first() or ''
        #content = response.xpath('//body//text()').extract()
        content = response.text.encode(response.encoding).decode('utf-8', errors='replace')


        # cleaned_content = self.clean_content(content)
        cleaned_content = content.strip()

        
        item = LinkedinSpiderItem()
        item['title'] = title
        item['content'] = cleaned_content  # Assign cleaned content
        
        return item if item and 'content' in item and item['content'] else {'title': '', 'content': ''}



    def save_to_html(self, response, file_path, main_page_url):
      # main_page_dir = main_page_url.replace('http://', '').replace('https://', '').replace('/', '_').replace('.', '_').replace('-', '_')
      parsed_url = urlparse(response.url)
      main_page_dir = parsed_url.netloc.replace('.', '_')  # Replace dots with underscores to avoid issues with file paths
      #change
      main_page_dir = os.path.join(f'/home/sxs7285/data/linkedin_prj/Initial_Linkedin_for_Scarping/Scrapped_data/HTMLs/HTML_{str(self.file_number)}', main_page_dir)
      # print("main_page_dir",main_page_dir)
      if not os.path.exists(main_page_dir):
          os.makedirs(main_page_dir)

      with open(os.path.join(main_page_dir, file_path), 'w', encoding='utf-8') as f:
          # print("os.path.join(main_page_dir, file_path)",os.path.join(main_page_dir, file_path))
          f.write(response.text)


    def clean_content(self, content_list):
        
        pass


    def save_to_json(self, url, content, link_extensions, pp_url):
            # Sanitize the URL to create a valid key (e.g., replace '/', '.', etc. with underscores)
            sanitized_url = url.replace('http://', '').replace('https://', '').replace('/', '_').replace('.', '_')
            main_sanitized_url = str(link_extensions)

            self.scraped_items[main_sanitized_url][sanitized_url] = {
                "url": url,
                "date_scraped": datetime.datetime.now().isoformat(),
                "title": content['title'],
                "content": content['content'],
                "pp_url": pp_url
            }

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(ModifiedPrivacyPolicySpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

      
    def handle_error(self, failure):
        # Capture the failed URL
        failed_url = failure.request.url

        # Initialize error details
        error_details = {
            "URL": failed_url,
            "HTTP Status": None,
            "Error Message": repr(failure)
        }

        # Handle different types of errors
        if failure.check(HttpError):
            # HttpError - get the response
            response = failure.value.response
            error_details["HTTP Status"] = response.status
            self.logger.error(f"HttpError occurred for URL: {failed_url}, Status: {response.status}")
            print(f"HttpError occurred for URL: {failed_url}, Status: {response.status}")
        elif failure.check(DNSLookupError):
            # DNS Lookup Error
            self.logger.error(f"DNSLookupError occurred for URL: {failed_url}")
            print(f"DNSLookupError occurred for URL: {failed_url}")

        elif failure.check(TimeoutError):
            # Timeout Error
            self.logger.error(f"TimeoutError occurred for URL: {failed_url}")
            print(f"TimeoutError occurred for URL: {failed_url}")

        else:
            # Other types of errors
            self.logger.error(f"Error occurred for URL: {failed_url}")
            print(f"Error occurred for URL: {failed_url}")

        # Write the error details to a CSV file
        #change
        with open(f'/home/sxs7285/data/linkedin_prj/Initial_Linkedin_for_Scarping/Scrapped_data/errors_{self.file_number}.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["URL", "HTTP Status", "Error Message"])
            
            writer.writerow(error_details)

def extract_filename(url):
    # Use regular expression to find the last segment of the URL
    match = re.search(r'/([^/]+)/?$', url)
    
    if match:
        filename = match.group(1)
        # Remove any invalid characters for file names (e.g., replace dots with underscores)
        clean_filename = re.sub(r'[^\w]', '_', filename)
        return clean_filename.lower()  # Convert to lowercase for consistency
    else:
        return None  # No valid filename found


def normalize_url(url):
    # Parse the URL
    parsed_url = urlparse(url)

    # Ensure the scheme is https
    scheme = 'https' if parsed_url.scheme != 'https' else parsed_url.scheme

    # Remove 'www.' if it exists
    netloc = parsed_url.netloc.replace('www.', '')

    # Rebuild the URL with the standardized format
    normalized = urlunparse((scheme, netloc, parsed_url.path.rstrip('/'), '', '', ''))

    return normalized

import scrapy


class LinkedinSpiderItem(scrapy.Item):
    content = scrapy.Field()
    item = scrapy.Field()
    title = scrapy.Field()  # Add this line

