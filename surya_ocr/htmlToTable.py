from bs4 import BeautifulSoup

def flatten_html(html_file, output_file):
    # Load the HTML content
    with open(html_file, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Iterate through each page and rearrange the text lines
    pages = soup.find_all("div", class_="page")
    current_top = 10  # Start from the top with some padding

    for page in pages:
        text_lines = page.find_all("div", class_="text-line")

        # Sort all text lines by their original `top` value to maintain logical order
        text_lines.sort(key=lambda text_line: float(text_line["style"].split("top:")[1].split("px")[0].strip()))

        # Rearrange all text lines into a single column
        for text_line in text_lines:
            # Update the position to create a single column
            style = text_line["style"]
            style = update_style(style, "left", "10px")  # Align to the left
            style = update_style(style, "top", f"{current_top}px")  # Place at the current top
            text_line["style"] = style

            # Increment the current top position by height + margin
            height = float(text_line["style"].split("height:")[1].split("px")[0].strip())
            current_top += height + 10  # Add a gap of 10px

    # Write the flattened HTML to the output file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(str(soup))

def update_style(style, key, value):
    """Update a specific style property in an inline style string."""
    styles = style.split(";")
    updated_styles = []
    for s in styles:
        if s.strip().startswith(key):
            updated_styles.append(f"{key}: {value}")
        elif s.strip():
            updated_styles.append(s.strip())
    return "; ".join(updated_styles) + ";"

# File paths
html_file = "output.html"  # Input HTML file
output_file = "flattened_output.html"  # Output HTML file

# Flatten the HTML
flatten_html(html_file, output_file)

print(f"Flattened HTML content written to {output_file}")
