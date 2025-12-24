# forecasting-backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from io import BytesIO
import base64
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ast  # TAMBAH INI untuk parse string list

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# Load model (sesuaikan path kalau perlu)
model = load_model('gru_sales_model.h5')

@app.route('/eda', methods=['POST'])
def eda():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    df = pd.read_csv(file)

    # EDA stats
    eda_results = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'summary': df.describe().to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'top_products': df.nlargest(3, 'jumlah_ulasan')[['nama', 'jumlah_ulasan']].to_dict('records') if 'jumlah_ulasan' in df.columns else [],
        'avg_rating_brand': df.groupby('brand')['rating'].mean().to_dict() if 'brand' in df.columns and 'rating' in df.columns else {},
    }

    # Graph 1: Distribusi Harga
    graphs = {}
    if 'harga' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['harga'], kde=True, color="indigo")
        plt.title("Distribusi Harga Produk")
        img1 = BytesIO()
        plt.savefig(img1, format='png')
        plt.close()
        img1.seek(0)
        graphs['graph_harga'] = base64.b64encode(img1.getvalue()).decode('utf-8')

    # Graph 2: Perbandingan Total Penjualan 3 Produk Top (berdasarkan total daily_sales)
    if 'daily_sales' in df.columns and 'nama' in df.columns:
        # Parse daily_sales string jadi list dan hitung total penjualan per produk
        df['total_sales'] = df['daily_sales'].apply(lambda x: sum(ast.literal_eval(x)))
        
        # Ambil 3 produk dengan total penjualan tertinggi
        top3_sales = df.nlargest(3, 'total_sales')[['nama', 'total_sales']]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='total_sales', y='nama', data=top3_sales, palette="Blues_d")
        plt.title("Perbandingan Total Penjualan 3 Produk Teratas")
        plt.xlabel("Total Unit Terjual")
        plt.ylabel("Nama Produk")
        img2 = BytesIO()
        plt.savefig(img2, format='png', bbox_inches='tight')
        plt.close()
        img2.seek(0)
        graphs['graph_top_sales'] = base64.b64encode(img2.getvalue()).decode('utf-8')

    # Graph 3: Distribusi Rating per Brand
    if 'brand' in df.columns and 'rating' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='brand', y='rating', data=df, palette="Set2")
        plt.title("Distribusi Rating per Brand")
        plt.xticks(rotation=45)
        img3 = BytesIO()
        plt.savefig(img3, format='png')
        plt.close()
        img3.seek(0)
        graphs['graph_rating_brand'] = base64.b64encode(img3.getvalue()).decode('utf-8')

    # Graph 4: Distribusi Motif (pie chart)
    if 'motif' in df.columns:
        motif_count = df['motif'].value_counts()
        plt.figure(figsize=(10, 6))
        plt.pie(motif_count, labels=motif_count.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
        plt.title("Distribusi Motif Produk")
        img4 = BytesIO()
        plt.savefig(img4, format='png')
        plt.close()
        img4.seek(0)
        graphs['graph_motif'] = base64.b64encode(img4.getvalue()).decode('utf-8')

    eda_results['graphs'] = graphs

    return jsonify(eda_results)

@app.route('/forecast', methods=['POST'])
def forecast():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    df = pd.read_csv(file)

    # Parse 'daily_sales' string jadi list float (karena di CSV berbentuk '[22, 11, ...]')
    if 'daily_sales' in df.columns:
        try:
            # Parse string list jadi array float
            daily_sales_str = df['daily_sales'].iloc[0]  # Asumsi 1 baris per produk, ambil baris pertama
            daily_sales = ast.literal_eval(daily_sales_str)  # Convert '[22, 11, ...]' jadi list
            data = np.array(daily_sales).reshape(-1, 1)  # Jadikan array 2D untuk scaler
        except Exception as e:
            return jsonify({"error": f"Gagal parse daily_sales: {str(e)}"}), 400
    else:
        return jsonify({"error": "Kolom 'daily_sales' tidak ditemukan"}), 400

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    window_size = 60  # Sesuaikan dengan model kamu
    X = []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size])

    X = np.array(X)
    if X.size == 0:
        return jsonify({"error": "Data terlalu pendek untuk window size 60"}), 400

    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    forecast_results = {
        'forecast': predictions.flatten().tolist(),
    }

    # Graph forecast
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Historical Daily Sales", color="blue")
    plt.plot(range(len(data), len(data) + len(predictions)), predictions, label="Forecast", color="green")
    plt.title("Forecasting Penjualan Harian")
    plt.xlabel("Hari")
    plt.ylabel("Penjualan")
    plt.legend()
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    forecast_results['graph'] = base64.b64encode(img.getvalue()).decode('utf-8')

    return jsonify(forecast_results)

if __name__ == '__main__':
    app.run(port=5000, debug=True)