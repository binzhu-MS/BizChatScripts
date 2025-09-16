#!/usr/bin/env python3
"""
Example Usage of Universal File Reader
=====================================

This script demonstrates how to use the UniversalFileReader for different file types
and how to integrate it with other processing workflows.

Usage Examples:
- Reading individual files
- Batch processing directories
- Integration with LLM processing
- Error handling and fallback methods
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import from the utils package (parent directory)
from utils.file_reader import UniversalFileReader


class FileProcessingExample:
    """Example class showing how to use UniversalFileReader in practice"""

    def __init__(self):
        self.reader = UniversalFileReader()
        print("📚 File Processing Example initialized")

    def read_single_file(self, file_path: str) -> Dict[str, Any]:
        """Example: Read a single file and return structured results

        Args:
            file_path: Path to the file to read

        Returns:
            Dict[str, Any]: Structured file content and metadata
        """
        print(f"\n📖 Reading single file: {file_path}")

        try:
            result = self.reader.read_file(file_path)

            if result["status"] == "success":
                print(f"✅ Successfully read {result['file_name']}")
                print(f"   Format: {result['file_extension']}")
                print(f"   Size: {result['file_size_mb']} MB")
                print(f"   Content length: {len(result['content'])} characters")
                print(f"   Method: {result.get('extraction_method', 'unknown')}")

                # Show format-specific info
                if "total_pages" in result:
                    print(f"   Pages: {result['total_pages']}")
                elif "total_sheets" in result:
                    print(f"   Sheets: {result['total_sheets']}")
                elif "slide_count" in result:
                    print(f"   Slides: {result['slide_count']}")

            else:
                print(f"❌ Failed to read {file_path}: {result['error']}")

            return result

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return {"status": "error", "error": str(e)}

    def process_directory(
        self, directory_path: str, output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Example: Process all supported files in a directory

        Args:
            directory_path: Directory containing files to process
            output_file: Optional JSON file to save results

        Returns:
            List[Dict[str, Any]]: Results for all processed files
        """
        print(f"\n📁 Processing directory: {directory_path}")

        if not os.path.exists(directory_path):
            print(f"❌ Directory not found: {directory_path}")
            return []

        # Find all supported files
        supported_extensions = set(self.reader.all_extensions)
        files_to_process = []

        for file_path in Path(directory_path).rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files_to_process.append(file_path)

        if not files_to_process:
            print(f"⚠️ No supported files found in {directory_path}")
            return []

        print(f"📋 Found {len(files_to_process)} supported files")

        # Process each file
        results = []
        successful = 0
        failed = 0

        for file_path in files_to_process:
            try:
                result = self.reader.read_file(str(file_path))
                results.append(result)

                if result["status"] == "success":
                    successful += 1
                    print(f"✅ {file_path.name} - {len(result['content'])} chars")
                else:
                    failed += 1
                    print(f"❌ {file_path.name} - {result['error']}")

            except Exception as e:
                failed += 1
                error_result = {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "status": "error",
                    "error": str(e),
                }
                results.append(error_result)
                print(f"❌ {file_path.name} - {e}")

        print(f"\n📊 Processing Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📁 Total: {len(files_to_process)}")

        # Save results if requested
        if output_file:
            self.save_results(results, output_file)
            print(f"💾 Results saved to: {output_file}")

        return results

    def save_results(self, results: List[Dict[str, Any]], output_file: str) -> None:
        """Save processing results to JSON file

        Args:
            results: Processing results
            output_file: Output file path
        """
        # Prepare results for JSON serialization (truncate very long content)
        serializable_results = []
        for result in results:
            clean_result = result.copy()

            # Truncate very long content for storage
            if "content" in clean_result and len(clean_result["content"]) > 5000:
                clean_result["content_preview"] = (
                    clean_result["content"][:5000] + "... (truncated)"
                )
                clean_result["content_length"] = len(clean_result["content"])
                del clean_result["content"]  # Remove full content to save space

            # Remove complex nested objects that may not serialize well
            if "sheets_data" in clean_result:
                clean_result["sheets_summary"] = list(
                    clean_result["sheets_data"].keys()
                )
                del clean_result["sheets_data"]

            serializable_results.append(clean_result)

        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    def content_for_llm_processing(self, file_path: str) -> str:
        """Example: Extract content optimized for LLM processing

        Args:
            file_path: Path to the file

        Returns:
            str: Clean content ready for LLM processing
        """
        print(f"\n🧠 Preparing content for LLM: {file_path}")

        result = self.reader.read_file(file_path)

        if result["status"] != "success":
            raise Exception(f"Could not read file: {result['error']}")

        content = result["content"]

        # Clean and format content for LLM
        # Remove excessive whitespace
        lines = [line.strip() for line in content.split("\n")]
        lines = [line for line in lines if line]  # Remove empty lines

        # Add document metadata as context
        context_header = f"""Document: {result['file_name']}
Format: {result['file_extension']}
Size: {result['file_size_mb']} MB
Extraction Method: {result.get('extraction_method', 'unknown')}

Content:
"""

        clean_content = context_header + "\n".join(lines)

        print(f"✅ Content prepared: {len(clean_content)} characters")
        return clean_content

    def demonstrate_capabilities(self) -> None:
        """Demonstrate the capabilities of the universal file reader"""
        print("\n🎯 Universal File Reader Capabilities Demonstration")
        print("=" * 60)

        # Show supported formats
        formats = self.reader.get_supported_formats()
        print(f"\n📋 Supported File Formats:")
        for format_type, extensions in formats.items():
            print(f"  {format_type.title()}: {', '.join(extensions)}")

        # Test capabilities
        print(f"\n🧪 Testing Available Libraries...")
        capabilities = self.reader.test_capabilities()

        format_status = {
            "PDF": capabilities["pdf_support"],
            "Word": capabilities["word_support"],
            "Excel": capabilities["excel_support"],
            "PowerPoint": capabilities["powerpoint_support"],
            "Text": capabilities["text_support"],
        }

        for format_name, supported in format_status.items():
            status = "✅ Available" if supported else "❌ Not Available"
            print(f"  {format_name}: {status}")

        print(
            f"\n📚 Available Libraries: {', '.join(capabilities['available_libraries'])}"
        )
        if capabilities["missing_libraries"]:
            print(
                f"⚠️ Missing Libraries: {', '.join(capabilities['missing_libraries'])}"
            )

        # Usage examples
        print(f"\n💡 Usage Examples:")
        print(f"  # Read a single file")
        print(f"  python universal_file_reader.py read_file --file_path='document.pdf'")
        print(f"  ")
        print(f"  # Get file information")
        print(
            f"  python universal_file_reader.py get_file_info --file_path='spreadsheet.xlsx'"
        )
        print(f"  ")
        print(f"  # Test capabilities")
        print(f"  python universal_file_reader.py test_capabilities")
        print(f"  ")
        print(f"  # In Python code:")
        print(f"  from universal_file_reader import UniversalFileReader")
        print(f"  reader = UniversalFileReader()")
        print(f"  result = reader.read_file('document.pdf')")


def main():
    """Main function demonstrating usage"""
    print("🚀 Universal File Reader - Usage Examples")
    print("=" * 50)

    # Create example processor
    processor = FileProcessingExample()

    # Demonstrate capabilities
    processor.demonstrate_capabilities()

    print(f"\n🎉 Universal File Reader is ready for use!")
    print(f"\nKey Benefits:")
    print(f"  ✅ Supports PDF, Word, Excel, PowerPoint, and Text files")
    print(f"  ✅ Multiple fallback methods for maximum compatibility")
    print(f"  ✅ Comprehensive error handling and logging")
    print(f"  ✅ Structured output perfect for LLM processing")
    print(f"  ✅ Fire CLI integration for command-line usage")
    print(f"  ✅ Easy integration with existing workflows")


if __name__ == "__main__":
    main()
