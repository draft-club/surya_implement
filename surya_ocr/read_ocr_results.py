import json
from bs4 import BeautifulSoup

# Function to check and adjust for overlapping text
def adjust_positions(positions, left, top, width, height):
    while True:
        overlap_detected = False
        for pos in positions:
            existing_left, existing_top, existing_width, existing_height = pos
            if not (top + height <= existing_top or top >= existing_top + existing_height or
                    left + width <= existing_left or left >= existing_left + existing_width):
                # If overlap is detected, shift the new text block downward
                top = existing_top + existing_height + 5  # Add 5px gap
                overlap_detected = True
                break
        if not overlap_detected:
            break
    positions.append((left, top, width, height))  # Save adjusted position
    return left, top

# Function to generate HTML
def json_to_html(json_input):
    html_content = """
    <html>
    <head>
        <title>Text Extraction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
            }
            .page {
                position: relative;
                width: 1190px; /* Set to match the page width */
                height: 1674px; /* Set to match the page height */
                margin: 20px auto;
                border: 1px solid #ddd;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            .document-title {
                position: absolute;
                top: 10px;
                left: 10px;
                font-size: 18px;
                font-weight: bold;
            }
            .text-line {
                position: absolute;
                font-size: 14px;
                white-space: nowrap;
            }
        </style>
    </head>
    <body>
    """

    for document_name, page_data in json_input.items():
        for page in page_data:
            html_content += f'<div class="page">\n'
            # Add document title at the top of the page
            html_content += f'<div class="document-title">{document_name}</div>\n'
            positions = []  # Track positions of text blocks to avoid overlap
            for line in page["text_lines"]:
                text = line.get("text", "")
                bbox = line.get("bbox", [0, 0, 0, 0])
                left = bbox[0]
                top = bbox[1]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                # Adjust position to prevent overlap
                left, top = adjust_positions(positions, left, top, width, height)

                # Add each text line as a div positioned using the adjusted bbox
                html_content += f'<div class="table" style="left: {left}px; top: {top}px; width: {width}px; height: {height}px;">{text}</div>\n'
            html_content += "</div>\n"

    html_content += """
    </body>
    </html>
    """
    return html_content

# def json_to_html(json_input):
#     html_content = """
#     <html>
#     <head>
#         <title>Text Extraction</title>
#         <style>
#             body {
#                 font-family: Arial, sans-serif;
#                 margin: 0;
#                 padding: 0;
#             }
#             .page {
#                 position: relative;
#                 width: 1190px; /* Set to match the page width */
#                 height: 1674px; /* Set to match the page height */
#                 margin: 20px auto;
#                 border: 1px solid #ddd;
#                 box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
#             }
#             .document-title {
#                 position: absolute;
#                 top: 10px;
#                 left: 10px;
#                 font-size: 18px;
#                 font-weight: bold;
#             }
#             .text-line {
#                 position: absolute;
#                 font-size: 14px;
#                 white-space: nowrap;
#             }
#         </style>
#     </head>
#     <body>
#     """
#
#     for document_name, page_data in json_input.items():
#         for page in page_data:
#             html_content += f'<div class="page">\n'
#             # Add document title at the top of the page
#             html_content += f'<div class="document-title">{document_name}</div>\n'
#             for line in page["text_lines"]:
#                 text = line.get("text", "")
#                 bbox = line.get("bbox", [0, 0, 0, 0])
#                 left = bbox[0]
#                 top = bbox[1]
#                 width = bbox[2] - bbox[0]
#                 height = bbox[3] - bbox[1]
#
#                 # Add each text line as a div positioned using the bbox
#                 html_content += f'<div class="text-line" style="left: {left}px; top: {top}px; width: {width}px; height: {height}px;">{text}</div>\n'
#             html_content += "</div>\n"
#
#     html_content += """
#     </body>
#     </html>
#     """
#     return html_content


# Generate HTML from the JSON data
input_file = "/home/ozdomar/Adil/doc_gpt/vision/surya_ocr/results/surya/results.json"  # Replace with your JSON file path
output_file = "output.html"

try:
    with open(input_file, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    # Generate HTML from the JSON data
    html_output = json_to_html(json_data)

    # Write the HTML output to a file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(html_output)

    print(f"HTML content written to {output_file}")

except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")
except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from the file {input_file}.")