from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import base64
from io import BytesIO

app = FastAPI()

# Serve static files (like CSS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load templates for HTML rendering
templates = Jinja2Templates(directory="templates")



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Initially display a placeholder image until the graph is generated
    return templates.TemplateResponse("index.html", {"request": request, "display_graph": False, "image": None})

@app.post("/generate", response_class=HTMLResponse)
async def generate(request: Request, choice: str = Form(...)):
    print(f"Received choice: {choice}")  # Debugging log

    try:
        # Define file paths
        old_file = f"OldData/{choice} Historical Data.csv"
        new_file = f"NewData/{choice} Historical Data.csv"

        # Check if files exist
        if not os.path.exists(old_file) or not os.path.exists(new_file):
            raise FileNotFoundError(f"Data files for {choice} are missing. Please check your directories.")

        # Load data
        df_new = pd.read_csv(new_file)
        df_old = pd.read_csv(old_file)

        # Process "Change %" column
        df_new["Change %"] = pd.to_numeric(df_new["Change %"].str.replace('%', ''), errors='coerce').fillna(0)
        df_old["Change %"] = pd.to_numeric(df_old["Change %"].str.replace('%', ''), errors='coerce').fillna(0)

        # Prepare data for processing
        full_data = np.array(df_old["Change %"])
        month_data = np.array(df_new["Change %"])
        MMean = np.mean(month_data)

        # Find the historical month closest to the new month's mean
        nearest_mean = np.inf
        idx = 0
        days_in_month = 30
        for i in range((len(full_data) // days_in_month) - 1):
            m_data = full_data[i * days_in_month: (i + 1) * days_in_month]
            mean_of_month = np.mean(m_data)
            if abs(MMean - mean_of_month) < abs(MMean - nearest_mean):
                nearest_mean = mean_of_month
                idx = i

        find_data = full_data[idx * days_in_month: (idx + 1) * days_in_month]

        # Apply KMeans clustering
        data = find_data.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(data)
        pca = PCA(n_components=1)
        reduced_data = pca.fit_transform(data)
        # Prepare graph data
        days = np.arange(1, len(find_data) + 1)
        df_plot = pd.DataFrame({"Day": days, "Change %": find_data, "Cluster": clusters})

        # Generate plot
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df_plot, x="Day", y="Change %", hue="Cluster", palette="coolwarm", s=100, edgecolor="black")
        plt.plot(days, find_data, linestyle='-', color='green', label='Daily Change %', linewidth=2)
        plt.title(f"Daily Change % with KMeans Clustering ({choice})")
        plt.xlabel("Day of the Month")
        plt.ylabel("Change %")
        plt.legend(title="Cluster")
        plt.grid(True)

        # Save plot to Base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close()

        # Return HTML with graph
        return templates.TemplateResponse("index.html", {
            "request": request,
            "display_graph": True,  # Indicate that the graph is ready
            "image": base64_image  # Set the graph image
        })

    except Exception as e:
        print(f"Error occurred: {e}")
        # Return error message and keep the placeholder image
        return templates.TemplateResponse("index.html", {
            "request": request,
            "display_graph": False,  # Keep the placeholder image if there's an error
            "image": None,
            "message": f"Error generating graph: {str(e)}"
        })
