import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import requests
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import h5py
import re
from underthesea import word_tokenize
from gensim import corpora, models, similarities
import pickle
import requests
import re
from io import StringIO

# Thiết lập cấu hình trang
st.set_page_config(
    page_title="Hệ thống Đề xuất Sản phẩm Shopee",
    page_icon="shopee.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_stop_words():
    STOP_WORD_FILE = 'vietnamese-stopwords.txt'
    with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
        stop_words = file.read()
    stop_words = stop_words.split('\n')
    return stop_words

@st.cache_resource
def load_dictionary(path='dictionary.dict'):
    return corpora.Dictionary.load(path)

@st.cache_resource
def load_tfidf_model(path='tfidf_model.tfidf'):
    return models.TfidfModel.load(path)

@st.cache_resource
def load_similarity_index(_tfidf_corpus, num_features):
    return similarities.SparseMatrixSimilarity(_tfidf_corpus, num_features=num_features)

@st.cache_resource
def load_content_gem(path='content_gem.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
@st.cache_resource
def load_baseline_model(path='baseline_model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_large_csv_from_gdrive():
    """
    Tải file CSV lớn từ Google Drive
    """
    # URL ban đầu 
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # Lấy session cookies
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # Kiểm tra xem có cần xác nhận không
    if "confirm" in response.text:
        confirm_match = re.search(r'confirm=([0-9A-Za-z]+)', response.text)
        if confirm_match:
            confirm_token = confirm_match.group(1)
            url = f"{url}&confirm={confirm_token}"
            response = session.get(url, stream=True)
    
    # Đọc nội dung
    content = response.content.decode('utf-8')
    
    # Chuyển đổi sang DataFrame
    df = pd.read_csv(StringIO(content))
    
    return df

@st.cache_resource
def load_pkl_from_gdrive(file_id):
    """
    Tải file pickle từ Google Drive và khôi phục đối tượng Python
    
    Tham số:
        file_id (str): ID của file trên Google Drive
    
    Trả về:
        object: Đối tượng Python được khôi phục từ file pickle
    """
    # URL dạng chia sẻ công khai của Google Drive
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        # Tải nội dung file
        response = requests.get(url)
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        
        # Sử dụng BytesIO để đọc dữ liệu nhị phân
        data_bytes = BytesIO(response.content)
        
        # Khôi phục đối tượng từ file pickle
        obj = pickle.load(data_bytes)
        
        return obj
    
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi khi tải file: {str(e)}")
        return None
    except pickle.UnpicklingError as e:
        st.error(f"Lỗi khi giải nén file pickle: {str(e)}")
        return None
        

@st.cache_data
def load_data():
    """
    Load và cache dữ liệu sản phẩm từ file CSV
    Returns:
        DataFrame: DataFrame chứa dữ liệu sản phẩm
    """
    try:
        df = pd.read_csv('Products_ThoiTrangNam_1.csv')
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file CSV: {e}")
        return pd.DataFrame()  # Trả về DataFrame trống nếu có lỗi
    
@st.cache_data
def load_data_user():
    """
    Load và cache dữ liệu sản phẩm từ file CSV
    Returns:
        DataFrame: DataFrame chứa dữ liệu sản phẩm
    """
    try:
        df = pd.read_csv('Products_ThoiTrangNam_rating.csv')
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file CSV: {e}")
        return pd.DataFrame()  # Trả về DataFrame trống nếu có lỗi
    
# Tải các tài nguyên sử dụng hàm đã được cache
stop_words = load_stop_words()
dictionary = load_dictionary()
tfidf = load_tfidf_model()
content_gem = load_content_gem()
feature_cnt = len(dictionary.token2id)
corpus = [dictionary.doc2bow(text) for text in content_gem]
tfidf_corpus = tfidf[corpus]
index = load_similarity_index(tfidf_corpus, feature_cnt)
products_df = load_data()
baseline_model = load_baseline_model()
user_df = load_large_csv_from_gdrive("1O-iWYMeHA2Epk5L3zg4UHYMJyqrjiI-pOkgIxnrj9dQ")

# 1. Sử dụng st.cache_data.clear() để xóa cache từ dữ liệu đã lưu trong đêcorator @st.cache_data
import streamlit as st

# if st.button("Xóa cache dữ liệu"):
#     st.cache_data.clear()
#     st.success("Đã xóa cache dữ liệu thành công!")

# # 2. Sử dụng st.cache_resource.clear() để xóa cache từ các tài nguyên đã lưu với @st.cache_resource
# if st.button("Xóa cache tài nguyên"):
#     st.cache_resource.clear()
#     st.success("Đã xóa cache tài nguyên thành công!")

# # 3. Xóa tất cả các loại cache
# if st.button("Xóa tất cả cache"):
#     st.cache_data.clear()
#     st.cache_resource.clear()
#     st.success("Đã xóa tất cả cache thành công!")
if 'option_products' not in st.session_state:
    # Giả sử bạn có DataFrame products_df và muốn lấy 10 sản phẩm ngẫu nhiên
    # Có thể thay đổi logic chọn ngẫu nhiên tùy thuộc vào yêu cầu cụ thể của bạn
    option_products = products_df['product_id'].sample(n=10)
    st.session_state['option_products'] = option_products


# Hàm mô phỏng để lấy dữ liệu sản phẩm
def get_sample_products(n=10):
    products = {
        'product_id': [f'P{i:04d}' for i in range(1, n+1)],
        'product_name': [f'Sản phẩm {i}' for i in range(1, n+1)],
        'category': np.random.choice(['Thời trang', 'Điện tử', 'Gia dụng', 'Mỹ phẩm', 'Đồ chơi'], n),
        'price': np.random.randint(100000, 2000000, n),
        'rating': np.random.uniform(3.5, 5.0, n).round(1),
        'image_url': [f'https://picsum.photos/id/{i+20}/200/200' for i in range(n)]
    }
    return pd.DataFrame(products)

# Hàm mô phỏng để lấy dữ liệu khách hàng
def get_sample_customers(n=5):
    customers = {
        'customer_id': [f'C{i:04d}' for i in range(1, n+1)],
        'name': [f'Khách hàng {i}' for i in range(1, n+1)]
    }
    return pd.DataFrame(customers)

# Hàm mô phỏng kết quả đã đạt được khi huấn luyện mô hình
def get_content_based_results():
    results = {
        'model': ['Gensim', 'Cosine_similarity'],
        'Độ chính xác': ['Cao', 'Cao'],
        'Thời gian dự đoán': ['Khá nhanh (0.1722s)', 'Rất nhanh (0.0035s)'],
        'Lượng bộ nhớ sử dụng': ['Khi huấn luyện tốn ít RAM hơn','Tốn nhiều RAM hơn'],
        'Similar_score': ['0.38-0.54', '0.36-0.56'],
    }
    return pd.DataFrame(results)


# Hàm mô phỏng để tìm sản phẩm tương tự dựa trên nội dung
def find_similar_products(description, products_df, n=10):
    # Trong thực tế, bạn sẽ sử dụng các vector đặc trưng từ mô hình đã huấn luyện
    # Đây chỉ là mô phỏng
    np.random.seed(hash(description) % 10000)
    similarity_scores = np.random.uniform(0.5, 1.0, len(products_df))
    products_df['similarity'] = similarity_scores
    return products_df.sort_values('similarity', ascending=False).head(n)

def is_valid_vietnamese(word):
    vietnamese_chars = (
        "a-zA-Z0-9_"
        "àáạảãâầấậẩẫăằắặẳẵ"
        "èéẹẻẽêềếệểễ"
        "ìíịỉĩ"
        "òóọỏõôồốộổỗơờớợởỡ"
        "ùúụủũưừứựửữ"
        "ỳýỵỷỹ"
        "đ"
        "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ"
        "ÈÉẸẺẼÊỀẾỆỂỄ"
        "ÌÍỊỈĨ"
        "ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
        "ÙÚỤỦŨƯỪỨỰỬỮ"
        "ỲÝỴỶỸ"
        "Đ"
    )
    pattern = f'^[{vietnamese_chars}]+$'
    return re.match(pattern, word) is not None

def get_similar_document_ids(input_string, dictionary, tfidf, index, stop_words, top_k=10):
    """
    Xử lý một chuỗi đầu vào, tìm các document tương tự và trả về list các ID.

    Args:
        input_string (str): Chuỗi đầu vào cần tìm kiếm sự tương tự.
        dictionary (corpora.Dictionary): Dictionary đã được tạo từ dữ liệu huấn luyện.
        tfidf (models.TfidfModel): Mô hình TF-IDF đã được huấn luyện.
        index (similarities.SparseMatrixSimilarity): Index tương tự đã được xây dựng.
        stop_words (set): Set chứa các stop words tiếng Việt.
        top_k (int): Số lượng document tương tự hàng đầu muốn trả về.

    Returns:
        list: Danh sách các ID của các document tương tự nhất.
    """
    if not isinstance(input_string, str):
        return []

    # 1. Kiểm tra tiếng Việt hợp lệ và loại bỏ các từ không hợp lệ (tùy chọn)
    words = input_string.split()
    valid_words = [w for w in words if is_valid_vietnamese(w)]
    processed_text = ' '.join(valid_words)
    print('dict', len(dictionary.token2id))

    # 2. Tiến hành word_tokenize
    tokenized_text = word_tokenize(processed_text, format="text")
    print('tokenized',tokenized_text)

    # 3. Loại bỏ stop_word
    filtered_tokens = [word.lower() for word in tokenized_text.split()]
    #print('filtered_tokens',filtered_tokens)
    # 4. Chuyển từ list các từ có nghĩa (đã loại bỏ stop words và viết thường)
    meaningful_words = [re.sub('[0-9]+', '', t) for t in filtered_tokens if t not in ['', ' ', ',', '.', '...', '-', ':', ';', '?', '%', '(', ')', '+', '/', "'", '&']]
    print('meaningful_words',meaningful_words)
    # 5. Sử dụng dictionary tạo trước đó để tạo kw_vector
    kw_vector = dictionary.doc2bow(meaningful_words)
    print('kw_vector',kw_vector)
    # 6. sim => đưa vào mô hình gensim đã tạo ở trên để dự đoán kết quả
    sim = index[tfidf[kw_vector]]
    print('sim',sim)
    sorted_similarities = sorted(enumerate(sim), key=lambda item: item[1], reverse=True)
    n = 0
    top_familier = 10
    familier_lst = []
    # In kết quả đã sắp xếp
    for doc_id, similarity in sorted_similarities:
        
        # Bỏ qua chính nó (tùy chọn, phụ thuộc vào mục đích sử dụng)
        if doc_id != 1:  # Có thể điều chỉnh điều kiện này tùy nhu cầu
            print("keyword is similar to doc_index %d: %.4f" % (doc_id, similarity))
            familier_lst.append(doc_id)
            n = n + 1
            if n>=top_familier:
                break
    print(familier_lst)
    return familier_lst

def display_product_cards(products, score_col=None):
    # Tạo 5 cột cho lưới sản phẩm
    cols = st.columns(5)
    
    # Dùng CSS riêng biệt thay vì HTML nội tuyến
    st.markdown("""
    <style>
        .product-card {
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
            height: 380px;
            position: relative;
            overflow: hidden;
            background-color: #121212;
        }
        .product-image {
            height: 180px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 10px;
        }
        .product-image img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .product-title {
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 8px;
            height: 40px;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }
        .product-category, .product-price, .product-similarity {
            font-size: 13px;
            margin-bottom: 5px;
        }
        .product-rating {
            position: absolute;
            bottom: 10px;
            left: 10px;
            color: gold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Hiển thị từng sản phẩm theo hàng và cột
    for i, (_, product) in enumerate(products.iterrows()):
        col_idx = i % 5
        with cols[col_idx]:
            # Sử dụng st.container để các phần tử không bị tách rời
            with st.container():
                # Tạo thẻ sản phẩm hoàn chỉnh
                card_html = f"""
                <div class="product-card">
                    <div class="product-image">
                        <img src="{product['image'] if not pd.isna(product['image']) and product['image'] is not None else 'no_image.png'}" alt="{product['product_name']}">
                    </div>
                    <div class="product-title">{product['product_name']}</div>
                    <div class="product-category">Danh mục: {product['sub_category']}</div>
                    <div class="product-price">Giá: {product['price']:,} đ</div>
                """
                
                # Thêm điểm tương đồng nếu có
                if score_col and score_col in product and not pd.isna(product[score_col]):
                    card_html += f'<div class="product-similarity">Độ tương đồng: {product[score_col]:.2f}</div>'
                
                # Thêm đánh giá sao
                if 'rating' in product and not pd.isna(product['rating']):
                    stars = "⭐" * int(product['rating'])
                    if product['rating'] % 1 >= 0.5:
                        stars += "✨"
                    card_html += f'<div class="product-rating">{stars}</div>'
                
                # Đóng thẻ sản phẩm
                card_html += "</div>"
                
                # Hiển thị thẻ sản phẩm
                st.markdown(card_html, unsafe_allow_html=True)

def get_user_recommendation(baseline_model,user_id,user_df,min_score=4,n=10):
    if user_id not in user_df['user_id'].unique():
        print(f"User ID {user_id} không tồn tại trong dữ liệu!")
        return []
    
    # Lấy danh sách các product_id mà người dùng đã đánh giá
    #print('user_df',user_df.columns)
    user_ratings = user_df[user_df['user_id'] == user_id]
    rated_products = user_ratings['product_id'].unique()
    
    # Lấy danh sách tất cả các product_id
    all_products = user_df['product_id'].unique()
    
    # Danh sách các product_id mà người dùng chưa đánh giá
    unrated_products = np.setdiff1d(all_products, rated_products)
    
    # Dự đoán đánh giá cho các sản phẩm chưa đánh giá
    predictions = []
    for product_id in unrated_products:
        predicted_rating = baseline_model.predict(user_id, product_id).est
        if predicted_rating >= min_score:
            predictions.append((product_id, predicted_rating))
    
    # Sắp xếp các dự đoán theo thứ tự giảm dần của đánh giá
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Trả về top-n đề xuất
    return predictions[:n]

def on_select_change():
    # Lấy giá trị hiện tại của selectbox từ session_state
    selected_value = st.session_state.product_selector
    
    # Thực hiện các hành động mong muốn với giá trị đã chọn
    st.session_state.last_selected = selected_value
    print(f"Đã chọn sản phẩm: {selected_value}")
    idx = products_df[products_df['product_id'] == int(selected_value)].index.tolist()
    print(idx)
    view_content= content_gem[idx[0]]
    kw_vector = dictionary.doc2bow(view_content)
    sim = index[tfidf[kw_vector]]
    sorted_similarities = sorted(enumerate(sim), key=lambda item: item[1], reverse=True)
    n = 0
    top_familier = 10
    familier_lst = []
    # In kết quả đã sắp xếp
    for doc_id, similarity in sorted_similarities:
        if doc_id != idx[0]:  # Có thể điều chỉnh điều kiện này tùy nhu cầu
            print("keyword is similar to doc_index %d: %.4f" % (doc_id, similarity))
            familier_lst.append(doc_id)
            n = n + 1
            if n>=top_familier:
                break
    st.session_state['similar_products_from_selectbox'] = products_df.iloc[familier_lst]



# Sidebar
st.sidebar.title("Hệ thống Đề xuất Sản phẩm")
st.sidebar.image("shopee_pic_1.jpg", width=250)
st.sidebar.markdown("---")

# Chọn trang trong sidebar
page = st.sidebar.selectbox(
    "Chọn chức năng:",
    ["Kết quả huấn luyện", "Tìm sản phẩm tương tự", "Đề xuất cá nhân hóa"]
)

# Tải dữ liệu mẫu
sample_products = get_sample_products(50)
sample_customers = get_sample_customers()
content_based_results = get_content_based_results()
results_df = pd.read_csv('cf_algorithms_results.csv')

# Trang 1: Kết quả huấn luyện
if page == "Kết quả huấn luyện":
    st.title("Kết quả Huấn luyện Mô hình Đề xuất")
    
    # Hiển thị thông tin tổng quan
    st.subheader("Thông tin tổng quan về Dự án")
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số sản phẩm", "46,000+", "")
    col2.metric("Tổng số người dùng", "650,000+", "")
    col3.metric("Tổng số đánh giá", "986,000+", "")
    
    st.markdown("---")
    
    # Hiển thị bảng kết quả
    st.subheader("Mô hình Content-based Filtering")
    # Sử dụng Styler.set_properties để highlight chính xác dòng thứ 2 (index 1)
    content_based_styled_df = content_based_results.style.apply(
    lambda x: ['background-color: lightgreen' if i==0 else '' 
              for i in range(len(content_based_results))],axis=0)

    st.table(content_based_styled_df)
    
    # Hiển thị biểu đồ so sánh các mô hình
    st.subheader("Mô hình User-based Filtering")
    user_based_styled_df = results_df.style.apply(
    lambda x: ['background-color: lightgreen' if i==8 else '' 
              for i in range(len(results_df))],axis=0)

    st.table(user_based_styled_df)
    

# Trang 2: Tìm sản phẩm tương tự
elif page == "Tìm sản phẩm tương tự":
    st.title("Tìm Sản phẩm Tương tự (Content-based)")
    
    # Nhập mô tả sản phẩm
    product_description = st.text_area(
        "Nhập mô tả sản phẩm hoặc từ khóa cần tìm:",
        height=100,
        placeholder="Ví dụ: áo ba lỗ, cà vạt, mắt kính, vớ, tất, áo khoác..."
    )
    
    st.info("👆 Vui lòng nhập mô tả sản phẩm bạn quan tâm để nhận đề xuất sản phẩm tương tự.")
    # Nút tìm kiếm
    search_btn = st.button("Tìm kiếm", type="primary")
    
    
    if search_btn and product_description:

        input_str = str(product_description)
        similar_products_list = get_similar_document_ids(input_str, dictionary, tfidf, index, stop_words, top_k=10)
        similar_products = products_df.iloc[similar_products_list]
        #print(similar_products[['product_id','image']])

        # Hiển thị kết quả
        st.subheader(f"Sản phẩm tương tự với mô tả của bạn:")
        st.info(f"Tìm thấy {len(similar_products)} sản phẩm tương tự dựa trên mô tả của bạn.")
        
        # Hiển thị sản phẩm dưới dạng thẻ
        display_product_cards(similar_products, score_col='similarity')
    
    # Hiển thị hướng dẫn nếu chưa nhập mô tả
    if not search_btn or not product_description:
        # Hiển thị một số sản phẩm phổ biến để tham khảo
        st.subheader("Một số sản phẩm phổ biến")
        valid_products = products_df[products_df['image'].notna()]
        popular_products = valid_products.sample(n=10)
        display_product_cards(popular_products,score_col='similarity')

        
    doc_id = st.selectbox(
            "Chọn sản phẩm cần tư vấn:",
            options=st.session_state['option_products'],
            format_func=lambda x: f"{x} - {products_df[products_df['product_id'] == x]['product_name'].values[0]}",
            on_change=on_select_change,
            key="product_selector"
        )
    if 'similar_products_from_selectbox' in st.session_state:
        st.subheader("Sản phẩm tương tự với sản phẩm bạn chọn:")
        display_product_cards(st.session_state['similar_products_from_selectbox'], score_col='similarity')

# Trang 3: Đề xuất cá nhân hóa
elif page == "Đề xuất cá nhân hóa":
    st.title("Đề xuất Sản phẩm Cá nhân hóa")
    
    # Tạo tab để chọn phương pháp đề xuất
    #tab1 = st.tabs(["Dựa trên người dùng"])
    
    # Tab 1: Đề xuất dựa trên ID người dùng
    #with tab1:
        # Chọn ID khách hàng từ danh sách
    customer_id = st.selectbox(
        "Chọn mã khách hàng:",
        options=user_df['user_id'].sample(n=10),
        format_func=lambda x: f"{x} - {user_df[user_df['user_id'] == x]['user'].values[0]}"
    )
    
    # Hoặc nhập ID khách hàng mới
    custom_id = st.text_input("Hoặc nhập mã khách hàng khác:")
    
    if custom_id:
        customer_id = int(custom_id)
        print('customer_id',customer_id)
    
    # Nút tìm kiếm
    rec_btn = st.button("Lấy đề xuất", key="rec_btn1", type="primary")
    
    if rec_btn and customer_id:
        # Lấy đề xuất sản phẩm dựa trên ID khách hàng
        productid_lst = []
        if customer_id not in user_df['user_id'].unique():
            print(user_df['user_id'].unique())
            st.info(f"User ID {customer_id} không tồn tại trong dữ liệu!")
        else:
            print(customer_id)
            top_recommendations = get_user_recommendation(baseline_model,customer_id,user_df)
            for i, (product_id, predicted_rating) in enumerate(top_recommendations, 1):
                productid_lst.append(product_id)
                df_recommend_list = products_df[products_df['product_id'].isin(productid_lst)]
            # Hiển thị tiêu đề với ID khách hàng
            st.subheader(f"Sản phẩm đề xuất cho khách hàng {customer_id}:")
            print(productid_lst)
            # Hiển thị sản phẩm đề xuất
            display_product_cards(df_recommend_list, score_col='recommendation_score')

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Hệ thống Đề xuất Sản phẩm Shopee </p>
        <p>© 2025 - Phát triển Cao Thị Ngọc Minh & Nguyễn Kế Nhựt</p>
    </div>
    """,
    unsafe_allow_html=True
)
