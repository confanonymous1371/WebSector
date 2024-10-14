

from transformers import Pipeline, RobertaForSequenceClassification, RobertaTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib import robotparser
from boilerpy3 import extractors
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from peft import PeftModel, PeftConfig
import torch
import numpy as np
import json
import os
from huggingface_hub import hf_hub_download

from datasets import load_dataset


class WebSectorPipeline(Pipeline):

    def __init__(self, model_name="Shahriar/WebSector-Flexible", device='cuda', kb_repo="Shahriar/PMI_analysis", mode="flexible", quantization_type=None):
        """
        Initialize the pipeline with a model and corresponding tokenizer. Choose the correct knowledge base 
        (either flexible or conservative) based on the mode. Optionally apply quantization.

        Args:
            model_name (str): Hugging Face model name (e.g., flexible or conservative).
            tokenizer: Tokenizer object for the model.
            device (str): Device to use ('cuda' or 'cpu').
            kb_repo (str): Hugging Face repository for the knowledge base.
            mode (str): Model mode, either 'flexible' or 'conservative' (default is 'flexible').
            quantization_type (str): Type of quantization ('4bit' or '8bit'). Default is None (no quantization).
        """
        self.model_name = model_name
        self.mode = mode  # Set the mode (flexible or conservative)
        self.kb_repo = kb_repo  # Repository for knowledge base
        self.quantization_type = quantization_type  # Quantization type ('4bit', '8bit', or None)


        self.model, self.tokenizer = self.load_model_from_hf()

        self.device = device
        # self.model.to(self.device)
        self.model.eval()
        
        # Load the appropriate knowledge base based on the mode
        self.knowledge_base = self.load_knowledge_base()
        super().__init__(model=self.model, tokenizer=self.tokenizer)


    def load_knowledge_base(self):
        """
        Load the knowledge base from Hugging Face based on the selected mode (flexible or conservative).
        
        Returns:
            dict: Knowledge base loaded from Hugging Face.
        """
        try:
            # Select the correct knowledge base file based on the mode
            if self.mode == "flexible":
                kb_path = load_dataset(self.kb_repo, data_files="knowledge_base_flexible.json", split="train")
            elif self.mode == "conservative":
                kb_path = load_dataset(self.kb_repo, data_files="knowledge_base_conservative.json", split="train")
            else:
                raise ValueError(f"Unsupported mode: {self.mode}. Choose 'flexible' or 'conservative'.")
            
            # Assuming kb_path contains the data in the correct format
            # print("Loaded knowledge base:", kb_path[0])
            return kb_path[0]
        
        except Exception as e:
            print(f"Failed to load knowledge base from Hugging Face: {str(e)}")
            return {}

    def load_model_from_hf(self):
        """
        Load the model and tokenizer from Hugging Face based on the model_name.
        Optionally apply quantization using BitsAndBytesConfig.

        Returns:
            model: Loaded Hugging Face model.
            tokenizer: Tokenizer for the model.
        """
        # Define quantization configuration if requested
        quantization_config = None
        if self.quantization_type in ['4bit', '8bit']:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True if self.quantization_type == '4bit' else False,
                load_in_8bit=True if self.quantization_type == '8bit' else False
            )

        # Load the model with or without quantization
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=10,
            quantization_config=quantization_config,  # Apply quantization if provided
            device_map="auto"  # Automatically map model to available devices (GPU/CPU)
        )

        # Load the tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(self.model_name)


        return model, tokenizer
    

    def generate_explanation(self, predicted_labels):
        """
        Generates a concise explanation based on PMI values and classification for the predicted labels on the validation set.

        Args:
            predicted_labels (list): A list of labels predicted by the model.

        Returns:
            explanations (list): A list of concise explanations based on PMI values and label relationships.
        """
        
        if not predicted_labels:
            return "No label was predicted."
        
        explanations = []
        n = len(predicted_labels)  # Number of predicted labels

        # Case where only one label is predicted
        if n == 1:
            predicted_label = predicted_labels[0]
            # print("predicted_label",predicted_label)
            # print("self.knowledge_base",self.knowledge_base)
            
            # Find all keys in the knowledge base that start with the predicted label
            keys1 = [key for key in self.knowledge_base if key.startswith(f"{predicted_label}__")]
            keys2 = [key for key in self.knowledge_base if key.endswith(f"{predicted_label}")]

            keys = list(set(keys1 +keys2))


            for key in keys:
                info = self.knowledge_base[key]

                # Only generate an explanation if there is an association (filter out 'No association')
                if info['Classification'] != 'No association':
                    # Split the key to get the less confident label
                    _, not_predicted_sector = key.split("__")
                    
                    # Create a concise explanation based on PMI and classification
                    explanation = (
                        f"On the validation set in '{self.mode}' mode, when {predicted_label}' is predicted, it apears to have a positive association with '{not_predicted_sector}' "
                        f"(The PMI is {round(info['PMI Value'], 2)}), indicating a '{info['Classification']}' relationship."
                        
                    )
                    # print("explanation",explanation)
                    explanations.append(explanation)
        
        # Case where more than one label is predicted
        else:
                    # for i in range(n):
                    #     # more_confident_label = predicted_labels[i]
                most_confident_label = predicted_labels[0]
                label_pair1 = [(most_confident_label,sector) for sector in predicted_labels]
                label_pair2 = [(sector, most_confident_label) for sector in predicted_labels]
                # label_pair = list(set(label_pair1 +label_pair2))
                label_pair = label_pair1 +label_pair2


                # print("label_pair",label_pair)
                # Compare the current label with other predicted labels
                for j in range(len(label_pair)):
                    # predicted_label = predicted_labels[j]
                    key = f"{label_pair[j][0]}__{label_pair[j][1]}"

                    keys1 = [key for key in self.knowledge_base if key.startswith(f"{label_pair[j][0]}__")]
                    keys2 = [key for key in self.knowledge_base if key.endswith(f"{label_pair[j][1]}")]

                    keys = list(set(keys1 +keys2))
                    # If the key exists in the knowledge base, generate an explanation
                    if key in self.knowledge_base:
                        info = self.knowledge_base[key]
                        # Only generate an explanation if there is an association (filter out 'No association')
                        if info['Classification'] != 'No association':
                            explanation = (
                                f"On the validation set in '{self.mode}' mode, '{label_pair[j][0]}' and '{label_pair[j][1]}' are predicted together, "
                                f"The PMI value between '{label_pair[j][0]}' and '{label_pair[j][1]}' is {round(info['PMI Value'], 2)}, indicating a '{info['Classification']}' relationship."
                            )
                            explanations.append(explanation)

                        elif info['Classification'] == 'No association': 

                            explanation = (
                                f"On the validation set in '{self.mode}' mode,"
                                f"The PMI value between '{label_pair[j][0]}' and '{label_pair[j][1]}' is {round(info['PMI Value'], 2)}, indicating a '{info['Classification']}' relationship."
                            )
                            explanations.append(explanation)
        
        

        return explanations


    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        if "sentences_count" in kwargs:
            preprocess_kwargs["sentences_count"] = kwargs["sentences_count"]
        if "threshold" in kwargs:
            postprocess_kwargs["threshold"] = kwargs["threshold"]
        if "return_content" in kwargs:
            postprocess_kwargs["return_content"] = kwargs["return_content"]
        return preprocess_kwargs, {}, postprocess_kwargs

    def preprocess(self, input_text, sentences_count=6):
        try:
            if input_text.startswith(('http://', 'https://',' http://', ' https://')):
                raw_text = self.scrape_website(input_text)
                extracted_text = self.extract_content(raw_text)
            else:
                extracted_text = input_text

            if not extracted_text:
                print("Warning: No content extracted. Using original input.")
                extracted_text = input_text

            summarized_text = self.summarize_text(extracted_text, sentences_count)

            if not summarized_text:
                print("Warning: Summarization failed. Using extracted text.")
                summarized_text = extracted_text  # Use first 1000 characters if summarization fails
            
            return {
                "summarized_text": summarized_text,
                "extracted_text": extracted_text,
                "original_input": input_text
            }
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return {
                "summarized_text": input_text,
                "extracted_text": input_text,
                "original_input": input_text
            }



    def scrape_website(self, url):
        try:
            parsed_url = urlparse(url)
            landing_page_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

            # Check robots.txt
            rp = robotparser.RobotFileParser()
            rp.set_url(urljoin(landing_page_url, "/robots.txt"))
            try:
                rp.read()
                if not rp.can_fetch("*", landing_page_url):
                    print(f"Robot.txt does not allow scraping for {landing_page_url}")
                    return ""  # Return empty string to indicate scraping not allowed
            except Exception as e:
                print(f"Error reading robots.txt for {landing_page_url}: {str(e)}")

            # Proceed with scraping if robots.txt allows
            response = requests.get(landing_page_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            main_content = soup.get_text()

            links = soup.find_all('a', href=True)
            allowed_links = [urljoin(landing_page_url, link['href']) for link in links if self.is_allowed_link(urljoin(landing_page_url, link['href']))]
            
            additional_content = ""
            for link in allowed_links[:5]:  # Limit the number of links to follow
                try:
                    if rp.can_fetch("*", link):
                        link_response = requests.get(link, timeout=5)
                        link_response.raise_for_status()
                        additional_content += link_response.text
                    else:
                        print(f"Robot.txt does not allow scraping for {link}")
                except requests.RequestException as e:
                    print(f"Error fetching {link}: {str(e)}")

            return main_content + " " + additional_content

        except requests.RequestException as e:
            print(f"Error fetching {url}: {str(e)}")
            return ""
        except Exception as e:
            print(f"An unexpected error occurred while processing {url}: {str(e)}")
            return ""


    def is_allowed_link(self, url):
        excluded_keywords = ['privacy', 'terms', 'conditions', 'policy']
        return not any(keyword in url.lower() for keyword in excluded_keywords)

    # def extract_content(self, raw_text):
    #     try:
    #         extractor = extractors.ArticleExtractor()
    #         content = extractor.get_content(raw_text)
    #         return content
    #     except Exception as e:
    #         print(f"Error extracting content with boilerpy3: {str(e)}")
    #         print("Falling back to BeautifulSoup for content extraction")
    #         try:
    #             soup = BeautifulSoup(raw_text, 'html.parser')
    #             # Remove script and style elements
    #             for script in soup(["script", "style"]):
    #                 script.decompose()
    #             # Get text
    #             text = soup.get_text()
    #             # Break into lines and remove leading and trailing space on each
    #             lines = (line.strip() for line in text.splitlines())
    #             # Break multi-headlines into a line each
    #             chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    #             # Drop blank lines
    #             text = '\n'.join(chunk for chunk in chunks if chunk)
    #             return text
    #         except Exception as e:
    #             print(f"Error extracting content with BeautifulSoup: {str(e)}")
    #             return raw_text  # Return the original text if all extraction methods fail
    

    def extract_content(self, raw_text):
        try:
            extractor = extractors.ArticleExtractor()
            content = extractor.get_content(raw_text)
            return content
        except Exception as e:
            # Handle boilerpy3 specific errors
            print(f"Error extracting content with boilerpy3: {str(e)}")
            print("Falling back to BeautifulSoup for content extraction")
            try:
                soup = BeautifulSoup(raw_text, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                # Get text and clean it
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                return text
            except Exception as e:
                # Final fallback to raw text if BeautifulSoup also fails
                print(f"Error extracting content with BeautifulSoup: {str(e)}")
                return raw_text  # Return raw text if everything else fails


    def summarize_text(self, text, sentences_count=6):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, sentences_count)
        return ' '.join(str(sentence) for sentence in summary)

    def _forward(self, model_inputs):
        # Tokenize the input text (summarized text)
        inputs = self.tokenizer(
            model_inputs["summarized_text"],
            return_tensors="pt",
            truncation=True,
            max_length=512,  # You can change the max length depending on the model's limits
            padding="max_length"
        )
        
        # Move inputs to the correct device (GPU/CPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get logits from the model output
        logits = outputs.logits
        
        # Apply sigmoid to convert logits to probabilities (multi-label classification)
        probabilities = torch.sigmoid(logits)

        # Threshold at 0.5 to determine the predicted classes (1 means class is predicted)
        predicted_classes = (probabilities > 0.5).int()

        return {
            "predicted_classes": predicted_classes[0].cpu().tolist(),  # Return as a list of predicted class labels
            "probabilities": probabilities[0].cpu().tolist(),  # Return as a list of probabilities
            "extracted_text": model_inputs["extracted_text"],  # Retain original text for postprocessing
            "summarized_text": model_inputs["summarized_text"]  # Retain summarized text for postprocessing
        }

    def postprocess(self, model_outputs, threshold=0.5, return_content=False, at_scale = False):
        # Define the class names that correspond to your model's output
        class_names = np.array([
            'civil, mechanical & electrical', 'consumer & supply chain', 'education',
            'finance, marketing & human resources', 'government, defense & legal',
            'information technology & electronics', 'medical', 'non-profit',
            'sports, media & entertainment', 'travel, food & hospitality'
        ])
        
        # Get the predicted probabilities and classes
        probabilities = np.array(model_outputs["probabilities"])
        predicted_classes = (probabilities > threshold).astype(int)
        
        # Get the predicted labels by finding the indices where predicted_classes == 1
        predicted_labels = list(class_names[np.where(predicted_classes == 1)[0]])

        # Generate explanations using the knowledge base
        explanations = self.generate_explanation(predicted_labels)
        
        # Construct the final result dictionary
        result = {
            "predicted_label": predicted_labels,  # List of predicted labels
            "all_probabilities": dict(zip(class_names, probabilities.tolist())),  # Map class names to their probabilities
            "threshold_used": threshold,  # The threshold used for classification
            "explanations": explanations  # Generated explanations from the knowledge base
        }

        # Optionally return the extracted and summarized content
        if return_content:
            result.update({
                "extracted_content": model_outputs["extracted_text"],
                "summarized_content": model_outputs["summarized_text"]
            })

        if  at_scale: 
           result = {
            'predicted_label': result['predicted_label'],
            'all_probabilities': result['all_probabilities']
            }
           
           



        return result



import aiohttp
import asyncio
import time
import psutil
import torch



import aiohttp
import asyncio
import time
import psutil
import torch

class BatchedWebSectorPipeline1(WebSectorPipeline):
    def __init__(self, *args, batch_size=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def get_system_utilization(self):
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = None
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.utilization(torch.cuda.current_device())
        return cpu_usage, gpu_usage

    # async def async_scrape_website(self, session, url):
    #     try:
    #         async with session.get(url, timeout=10) as response:
    #             if response.status == 200:
    #                 html_content = await response.text()
    #                 return html_content
    #             else:
    #                 print(f"Failed to scrape {url}: {response.status}")
    #                 return ""
    #     except Exception as e:
    #         print(f"Error scraping {url}: {e}")
    #         return ""

    # async def scrape_websites(self, urls):
    #     async with aiohttp.ClientSession() as session:
    #         tasks = [self.async_scrape_website(session, url) for url in urls]
    #         return await asyncio.gather(*tasks)  # Run all tasks concurrently


    async def async_scrape_website(self, session, url):
        max_retries = 3  # Retry up to 3 times
        for attempt in range(max_retries):
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        print(f"Failed to scrape {url}: {response.status}")
                        return ""
            except Exception as e:
                print(f"Error scraping {url}, attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    return ""

    async def scrape_websites(self, urls):
        async with aiohttp.ClientSession() as session:
            tasks = [self.async_scrape_website(session, url) for url in urls]
            return await asyncio.gather(*tasks)

    def preprocess_batch(self, inputs, sentences_count=6):
        start_time = time.time()

        # Step 1: Scrape websites asynchronously
        scraping_start = time.time()
        # scraped_contents = asyncio.run(self.scrape_websites(inputs))

        scraped_contents = asyncio.get_event_loop().run_until_complete(self.scrape_websites(inputs)) 
        scraping_end = time.time()

        # Measure CPU and GPU utilization after web scraping
        cpu_scraping, gpu_scraping = self.get_system_utilization()

        # Step 2: Process the scraped content (e.g., summarizing)
        summarization_start = time.time()
        preprocessed_data = []
        for i, content in enumerate(scraped_contents):
            if not content:
                print(f"Warning: No content extracted for {inputs[i]}. Using original input.")
                content = inputs[i]
            summarized_text = self.summarize_text(content, sentences_count)
            preprocessed_data.append({
                "summarized_text": summarized_text,
                "extracted_text": content,
                "original_input": inputs[i]
            })
        summarization_end = time.time()

        # Measure CPU and GPU utilization after summarization
        cpu_summarization, gpu_summarization = self.get_system_utilization()

        # Timing for preprocessing
        total_preprocess_time = time.time() - start_time

        # Store times and system utilization for each stage
        self.timing_info = {
            "scraping_time": scraping_end - scraping_start,
            "summarization_time": summarization_end - summarization_start,
            "total_preprocess_time": total_preprocess_time,
            "cpu_scraping": cpu_scraping,
            "gpu_scraping": gpu_scraping,
            "cpu_summarization": cpu_summarization,
            "gpu_summarization": gpu_summarization,
        }

        return preprocessed_data

    def process_batch(self, inputs, sentences_count=6, threshold=0.5, return_content=False, at_scale = True):
        start_time = time.time()

        # Preprocess the batch
        model_inputs = self.preprocess_batch(inputs, sentences_count)

        # Forward pass in a batch (tokenization and model inference)
        prediction_start = time.time()
        model_outputs = self._forward_batch(model_inputs)
        prediction_end = time.time()

        # Measure CPU and GPU utilization after prediction
        cpu_prediction, gpu_prediction = self.get_system_utilization()

        # Postprocess the outputs to get predictions and explanations
        postprocess_start = time.time()
        results = self.postprocess_batch(model_outputs, threshold, return_content, at_scale)
        postprocess_end = time.time()

        # Final system utilization
        cpu_postprocess, gpu_postprocess = self.get_system_utilization()

        total_time = time.time() - start_time

        # Log the timing and system utilization for each stage
        self.timing_info.update({
            "prediction_time": prediction_end - prediction_start,
            "postprocess_time": postprocess_end - postprocess_start,
            "total_time": total_time,
            "cpu_prediction": cpu_prediction,
            "gpu_prediction": gpu_prediction,
            "cpu_postprocess": cpu_postprocess,
            "gpu_postprocess": gpu_postprocess,
        })

        # Optionally print or return timing info for debugging
        self.print_timing_info()

        return results, self.timing_info

    def _forward_batch(self, model_inputs):
        # Tokenize all inputs in a batch using Hugging Face's tokenizer
        inputs = self.tokenizer(
            [mi["summarized_text"] for mi in model_inputs],  # Extract the summarized texts
            return_tensors="pt",
            truncation=True,
            padding=True  # Pad dynamically
        )

        # Move inputs to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass (batch inference)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(outputs.logits)

        # Collect outputs for each input
        return [
            {
                "probabilities": probabilities[i].cpu().tolist(),
                "extracted_text": model_inputs[i]["extracted_text"],
                "summarized_text": model_inputs[i]["summarized_text"]
            }
            for i in range(len(model_inputs))
        ]

    def postprocess_batch(self, model_outputs, threshold=0.5, return_content=False, at_scale = True):
        # Post-process each model output in the batch
        results = [self.postprocess(output, threshold, return_content, at_scale) for output in model_outputs]
        print("results",results)
        return results

    def print_timing_info(self):
        # Print the timing and system utilization for each process
        print("Timing Information:")
        for key, value in self.timing_info.items():
            print(f"{key}: {value}")
