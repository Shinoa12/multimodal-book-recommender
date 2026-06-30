from IPython.display import HTML, display
import pandas as pd

def display_recommendations(results):
    """
    Displays the recommendation results in a user-friendly format.

    Args:
        results (pd.DataFrame): DataFrame containing the recommendations with scores.
    """
    if results.empty:
        print("No recommendations to display.")
        return
    
    html = ""

    print(f"\n--- Displaying Top {len(results)} Recommendations ---")
    for _, row in results.iterrows():
        image_html = (
            f'<img src="{row["image_url"]}" width="100">'
            if pd.notnull(row["image_url"])
            else "No Image"
        )

        score_html = f"<b>Visual Score:</b> {row['visual_score']:.4f}<br>"

        if "text_score" in results.columns:
            score_html += f"<b>Text Score:</b> {row['text_score']:.4f}<br>"

        if "multimodal_score" in results.columns:
            score_html += f"<b>Multimodal Score:</b> {row['multimodal_score']:.4f}<br>"

        html += f"""
        <div style="border:1px solid #ccc; padding:10px; margin-bottom:10px;
                    display:flex; align-items:center; border-radius:8px;">
            <div style="margin-right:15px;">{image_html}</div>
            <div>
                <b>Title:</b> {row['titulo']}<br>
                <b>Authors:</b> {row['authors']}<br>
                {score_html}
            </div>
        </div>
        """
    display(HTML(html))


def display_recommendations_text(results):
    """
    Displays the recommendation results in a user-friendly text format.

    Args:
        results (pd.DataFrame): DataFrame containing the recommendations with scores.
    """
    if results.empty:
        print("No recommendations to display.")
        return

    print(f"\n--- Displaying Top {len(results)} Recommendations ---")
    for _, row in results.iterrows():
        print(f"Title: {row['titulo']}")
        print(f"Authors: {row['authors']}")
        print(f"Visual Score: {row['visual_score']:.4f}")
        
        if "text_score" in results.columns:
            print(f"Text Score: {row['text_score']:.4f}")
        
        if "multimodal_score" in results.columns:
            print(f"Multimodal Score: {row['multimodal_score']:.4f}")
        
        print("-" * 40)