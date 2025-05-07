#!/usr/bin/env python
"""
Model Cache Management Tool

This script provides command-line utilities for managing the model cache.
"""

import os
import argparse
import logging

from src.utils.model_cache import ModelCache
from src.utils.cache_utils import list_cached_models, print_cache_summary, delete_model_from_cache


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main():
    """Execute the cache management tool."""
    parser = argparse.ArgumentParser(description="Model Cache Management Tool")
    
    # Setup cache directory argument
    parser.add_argument(
        "--cache-dir", 
        default=os.path.join(os.path.dirname(__file__), "model_cache"),
        help="Directory where models are cached"
    )
    
    # Setup subparsers
    subparsers = parser.add_subparsers(dest='command', help='Cache management commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all cached models')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear all cached models')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a specific model from cache')
    delete_parser.add_argument('model_hash', help='Hash of the model to delete')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show detailed information about the cache')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Create cache directory if it doesn't exist
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Execute command
    if args.command == 'list':
        print_cache_summary(args.cache_dir)
    
    elif args.command == 'clear':
        cache = ModelCache(args.cache_dir)
        cache.clear_cache()
        logging.info(f"Cache cleared: {args.cache_dir}")
    
    elif args.command == 'delete':
        success = delete_model_from_cache(args.model_hash, args.cache_dir)
        if success:
            logging.info(f"Model {args.model_hash} deleted from cache")
        else:
            logging.error(f"Failed to delete model {args.model_hash}")
    
    elif args.command == 'info':
        cache = ModelCache(args.cache_dir)
        models = list_cached_models(args.cache_dir)
        
        # Calculate storage usage
        total_size = 0
        for model_hash in cache.metadata:
            model_path = cache.get_model_path(model_hash)
            if os.path.exists(model_path):
                total_size += sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(model_path)
                    for filename in filenames
                )
        
        # Print cache information
        print(f"Cache directory: {args.cache_dir}")
        print(f"Total models: {len(models)}")
        print(f"Total storage: {total_size / (1024*1024):.2f} MB")
        
    else:
        # No command specified, show help
        parser.print_help()


if __name__ == "__main__":
    main()