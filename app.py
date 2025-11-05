import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(
    page_title="Takeshy Velasquez Diaz",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .stApp {
        background: transparent;
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(45deg, #00f5ff, #00d4ff, #00b8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.3);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(0, 245, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 245, 255, 0.15);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 245, 255, 0.3);
        border: 1px solid rgba(0, 245, 255, 0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        color: #00f5ff;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        font-size: 18px;
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #00f5ff, #00d4ff);
        color: #0f0c29;
        border: 1px solid #00f5ff;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.5);
    }
    
    .dataframe {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border-radius: 10px;
        font-family: 'Rajdhani', sans-serif;
    }
    
    .element-container {
        font-family: 'Rajdhani', sans-serif;
        color: #e0e0e0;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #00f5ff, #00d4ff);
        color: #0f0c29;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 245, 255, 0.4);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 25px rgba(0, 245, 255, 0.6);
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        color: #00f5ff;
        font-size: 32px;
    }
    
    div[data-testid="stMetricLabel"] {
        font-family: 'Rajdhani', sans-serif;
        color: #a0a0a0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 48px;'>🧪 ML DATA PREPROCESSING LAB</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0a0a0; font-family: Rajdhani; font-size: 20px;'>Procesamiento Avanzado de Datasets para Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("<h2 style='font-size: 28px;'>⚙️ CONTROL PANEL</h2>", unsafe_allow_html=True)
    dataset_option = st.selectbox(
        "Selecciona Dataset",
        ["🚢 Titanic", "📚 Student Performance", "🌸 Iris"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Parámetros")
    test_size = st.slider("Tamaño Test (%)", 10, 50, 30, 5)
    random_state = st.number_input("Random State", 0, 100, 42)
    st.markdown("</div>", unsafe_allow_html=True)

def show_metrics(col1_text, col1_val, col2_text, col2_val, col3_text, col3_val):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(col1_text, col1_val)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(col2_text, col2_val)
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(col3_text, col3_val)
        st.markdown("</div>", unsafe_allow_html=True)

if dataset_option == "🚢 Titanic":
    st.markdown("<h2>🚢 ANÁLISIS DATASET TITANIC</h2>", unsafe_allow_html=True)
    
    st.info("📌 Puedes cargar tu propio archivo titanic.csv o usar datos de ejemplo")
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=['csv'])
    
    if uploaded_file is not None or st.button("Usar datos de ejemplo Titanic"):
        if uploaded_file is None:
            np.random.seed(42)
            df = pd.DataFrame({
                'PassengerId': range(1, 201),
                'Survived': np.random.choice([0, 1], 200, p=[0.6, 0.4]),
                'Pclass': np.random.choice([1, 2, 3], 200),
                'Name': [f'Passenger {i}' for i in range(200)],
                'Sex': np.random.choice(['male', 'female'], 200),
                'Age': np.random.normal(29, 14, 200).clip(0.5, 80),
                'SibSp': np.random.poisson(0.5, 200),
                'Parch': np.random.poisson(0.4, 200),
                'Ticket': [f'T{i}' for i in range(200)],
                'Fare': np.random.gamma(2, 15, 200),
                'Cabin': [f'C{i}' if i % 3 == 0 else np.nan for i in range(200)],
                'Embarked': np.random.choice(['S', 'C', 'Q'], 200)
            })
            df.loc[np.random.choice(df.index, 20), 'Age'] = np.nan
            df.loc[np.random.choice(df.index, 5), 'Embarked'] = np.nan
        else:
            df = pd.read_csv(uploaded_file)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📥 Carga", "🔍 Exploración", "🧹 Limpieza", "⚙️ Procesamiento", "📊 Resultados"])
        
        with tab1:
            st.markdown("### 📥 DATOS ORIGINALES")
            show_metrics("Total Registros", len(df), "Columnas", len(df.columns), "Tamaño (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab2:
            st.markdown("### 🔍 EXPLORACIÓN INICIAL")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Información del Dataset")
                buffer = StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
            
            with col2:
                st.markdown("#### Valores Nulos")
                null_data = df.isnull().sum()
                null_df = pd.DataFrame({
                    'Columna': null_data.index,
                    'Nulos': null_data.values,
                    'Porcentaje': (null_data.values / len(df) * 100).round(2)
                })
                st.dataframe(null_df[null_df['Nulos'] > 0], use_container_width=True)
            
            st.markdown("#### Estadísticas Descriptivas")
            st.dataframe(df.describe(), use_container_width=True)
        
        with tab3:
            st.markdown("### 🧹 LIMPIEZA DE DATOS")
            
            df_clean = df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], errors='ignore')
            st.success("✅ Columnas eliminadas: Name, Ticket, Cabin, PassengerId")
            
            if 'Age' in df_clean.columns:
                df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
                st.success(f"✅ Age: nulos rellenados con mediana ({df_clean['Age'].median():.1f})")
            
            if 'Embarked' in df_clean.columns:
                df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
                st.success(f"✅ Embarked: nulos rellenados con moda ({df_clean['Embarked'].mode()[0]})")
            
            if 'Fare' in df_clean.columns:
                df_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)
            
            duplicates = df_clean.duplicated().sum()
            df_clean = df_clean.drop_duplicates()
            st.success(f"✅ Duplicados eliminados: {duplicates}")
            
            show_metrics("Registros Finales", len(df_clean), "Nulos Restantes", df_clean.isnull().sum().sum(), "Columnas", len(df_clean.columns))
            
            st.dataframe(df_clean.head(), use_container_width=True)
        
        with tab4:
            st.markdown("### ⚙️ CODIFICACIÓN Y NORMALIZACIÓN")
            
            df_processed = df_clean.copy()
            
            if 'Sex' in df_processed.columns:
                le = LabelEncoder()
                df_processed['Sex'] = le.fit_transform(df_processed['Sex'])
                st.success("✅ Sex codificado: male=1, female=0")
            
            if 'Embarked' in df_processed.columns:
                df_processed = pd.get_dummies(df_processed, columns=['Embarked'], prefix='Embarked')
                st.success("✅ Embarked: One-Hot Encoding aplicado")
            
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            if 'Survived' in numeric_cols:
                numeric_cols.remove('Survived')
            
            scaler = StandardScaler()
            df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
            st.success(f"✅ Variables estandarizadas: {', '.join(numeric_cols)}")
            
            st.dataframe(df_processed.head(), use_container_width=True)
        
        with tab5:
            st.markdown("### 📊 DIVISIÓN Y RESULTADOS FINALES")
            
            if 'Survived' in df_processed.columns:
                X = df_processed.drop('Survived', axis=1)
                y = df_processed['Survived']
            else:
                X = df_processed
                y = None
            
            if y is not None:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=random_state
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🎯 Conjunto de Entrenamiento")
                    show_metrics("Muestras", X_train.shape[0], "Features", X_train.shape[1], "% del Total", f"{(1-test_size/100)*100:.0f}%")
                    st.dataframe(X_train.head(), use_container_width=True)
                
                with col2:
                    st.markdown("#### 🧪 Conjunto de Prueba")
                    show_metrics("Muestras", X_test.shape[0], "Features", X_test.shape[1], "% del Total", f"{test_size}%")
                    st.dataframe(X_test.head(), use_container_width=True)
                
                st.markdown("#### 📈 Visualización de Distribución")
                fig = go.Figure()
                fig.add_trace(go.Bar(x=['Entrenamiento', 'Prueba'], 
                                    y=[len(X_train), len(X_test)],
                                    marker_color=['#00f5ff', '#ff6b9d']))
                fig.update_layout(
                    title="Distribución Train-Test",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e0e0', family='Rajdhani')
                )
                st.plotly_chart(fig, use_container_width=True)

elif dataset_option == "📚 Student Performance":
    st.markdown("<h2>📚 ANÁLISIS STUDENT PERFORMANCE</h2>", unsafe_allow_html=True)
    
    st.info("📌 Puedes cargar tu propio archivo student-mat.csv o usar datos de ejemplo")
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=['csv'])
    
    if uploaded_file is not None or st.button("Usar datos de ejemplo Student"):
        if uploaded_file is None:
            np.random.seed(42)
            n = 200
            df = pd.DataFrame({
                'school': np.random.choice(['GP', 'MS'], n),
                'sex': np.random.choice(['M', 'F'], n),
                'age': np.random.randint(15, 23, n),
                'address': np.random.choice(['U', 'R'], n),
                'famsize': np.random.choice(['LE3', 'GT3'], n),
                'Pstatus': np.random.choice(['T', 'A'], n),
                'Medu': np.random.randint(0, 5, n),
                'Fedu': np.random.randint(0, 5, n),
                'studytime': np.random.randint(1, 5, n),
                'failures': np.random.randint(0, 4, n),
                'schoolsup': np.random.choice(['yes', 'no'], n),
                'famsup': np.random.choice(['yes', 'no'], n),
                'paid': np.random.choice(['yes', 'no'], n),
                'activities': np.random.choice(['yes', 'no'], n),
                'higher': np.random.choice(['yes', 'no'], n),
                'internet': np.random.choice(['yes', 'no'], n),
                'romantic': np.random.choice(['yes', 'no'], n),
                'absences': np.random.randint(0, 50, n),
                'G1': np.random.randint(0, 21, n),
                'G2': np.random.randint(0, 21, n),
            })
            df['G3'] = (df['G1'] * 0.3 + df['G2'] * 0.4 + np.random.randint(-3, 4, n)).clip(0, 20)
        else:
            df = pd.read_csv(uploaded_file)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📥 Carga", "🔍 Exploración", "🧹 Limpieza", "⚙️ Procesamiento", "📊 Resultados"])
        
        with tab1:
            st.markdown("### 📥 DATOS ORIGINALES")
            show_metrics("Total Registros", len(df), "Columnas", len(df.columns), "Variables Cat.", len(df.select_dtypes(include='object').columns))
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab2:
            st.markdown("### 🔍 EXPLORACIÓN INICIAL")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Variables Categóricas")
                cat_cols = df.select_dtypes(include='object').columns.tolist()
                st.write(", ".join(cat_cols))
                
                st.markdown("#### Variables Numéricas")
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                st.write(", ".join(num_cols))
            
            with col2:
                st.markdown("#### Valores Nulos")
                null_data = df.isnull().sum()
                st.write(f"Total nulos: {null_data.sum()}")
                if null_data.sum() > 0:
                    st.dataframe(null_data[null_data > 0])
            
            st.markdown("#### Estadísticas")
            st.dataframe(df.describe(), use_container_width=True)
        
        with tab3:
            st.markdown("### 🧹 LIMPIEZA DE DATOS")
            
            df_clean = df.drop_duplicates()
            st.success(f"✅ Duplicados eliminados: {len(df) - len(df_clean)}")
            
            nulls = df_clean.isnull().sum().sum()
            if nulls > 0:
                df_clean = df_clean.dropna()
                st.success(f"✅ Filas con nulos eliminadas: {nulls}")
            else:
                st.success("✅ No se encontraron valores nulos")
            
            show_metrics("Registros Finales", len(df_clean), "Columnas", len(df_clean.columns), "Nulos", df_clean.isnull().sum().sum())
            st.dataframe(df_clean.head(), use_container_width=True)
        
        with tab4:
            st.markdown("### ⚙️ CODIFICACIÓN Y NORMALIZACIÓN")
            
            df_processed = df_clean.copy()
            
            cat_cols = df_processed.select_dtypes(include='object').columns.tolist()
            if cat_cols:
                df_processed = pd.get_dummies(df_processed, columns=cat_cols, drop_first=True)
                st.success(f"✅ One-Hot Encoding aplicado a: {', '.join(cat_cols)}")
            
            numeric_cols = ['age', 'absences', 'G1', 'G2']
            numeric_cols = [col for col in numeric_cols if col in df_processed.columns]
            
            scaler = MinMaxScaler()
            df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
            st.success(f"✅ Variables normalizadas (0-1): {', '.join(numeric_cols)}")
            
            st.dataframe(df_processed.head(10), use_container_width=True)
        
        with tab5:
            st.markdown("### 📊 ANÁLISIS DE CORRELACIÓN Y RESULTADOS")
            
            if all(col in df_clean.columns for col in ['G1', 'G2', 'G3']):
                st.markdown("#### 🔗 Correlación entre Notas")
                corr_matrix = df_clean[['G1', 'G2', 'G3']].corr()
                
                fig = px.imshow(corr_matrix, 
                               text_auto='.2f',
                               color_continuous_scale='Blues',
                               title="Matriz de Correlación G1, G2, G3")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e0e0')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if 'G3' in df_processed.columns:
                X = df_processed.drop('G3', axis=1)
                y = df_processed['G3']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=random_state
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🎯 Conjunto de Entrenamiento")
                    show_metrics("Muestras", X_train.shape[0], "Features", X_train.shape[1], "% del Total", f"{(1-test_size/100)*100:.0f}%")
                
                with col2:
                    st.markdown("#### 🧪 Conjunto de Prueba")
                    show_metrics("Muestras", X_test.shape[0], "Features", X_test.shape[1], "% del Total", f"{test_size}%")
                
                st.markdown("#### 📈 Distribución de Nota Final (G3)")
                fig = px.histogram(df_clean, x='G3', nbins=20, 
                                  title="Distribución de Notas Finales",
                                  color_discrete_sequence=['#00f5ff'])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e0e0')
                )
                st.plotly_chart(fig, use_container_width=True)

else:  
    st.markdown("<h2>🌸 ANÁLISIS DATASET IRIS</h2>", unsafe_allow_html=True)
    
    if st.button("Cargar Dataset Iris desde sklearn"):
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        tab1, tab2, tab3, tab4 = st.tabs(["📥 Carga", "🔍 Exploración", "⚙️ Procesamiento", "📊 Visualización"])
        
        with tab1:
            st.markdown("### 📥 DATASET IRIS")
            show_metrics("Total Registros", len(df), "Features", len(df.columns)-2, "Clases", df['target'].nunique())
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("#### Información del Dataset")
            st.write(iris.DESCR[:500] + "...")
        
        with tab2:
            st.markdown("### 🔍 EXPLORACIÓN")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Estadísticas por Clase")
                st.dataframe(df.groupby('species').mean(), use_container_width=True)
            
            with col2:
                st.markdown("#### Distribución de Clases")
                class_dist = df['species'].value_counts()
                fig = px.pie(values=class_dist.values, names=class_dist.index,
                            title="Distribución de Especies",
                            color_discrete_sequence=['#00f5ff', '#ff6b9d', '#ffd700'])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e0e0')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Estadísticas Descriptivas")
            st.dataframe(df.describe(), use_container_width=True)
        
        with tab3:
            st.markdown("### ⚙️ ESTANDARIZACIÓN Y DIVISIÓN")
            
            X = df[iris.feature_names]
            y = df['target']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            df_scaled = pd.DataFrame(X_scaled, columns=iris.feature_names)
            df_scaled['target'] = y.values
            df_scaled['species'] = df['species'].values
            
            st.success("✅ Estandarización aplicada con StandardScaler")
            st.dataframe(df_scaled.head(10), use_container_width=True)
            
            st.markdown("#### Estadísticas Después de Estandarización")
            st.dataframe(df_scaled[iris.feature_names].describe(), use_container_width=True)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size/100, random_state=random_state
            )
            
            show_metrics("Train Shape", f"{X_train.shape[0]}x{X_train.shape[1]}", 
                        "Test Shape", f"{X_test.shape[0]}x{X_test.shape[1]}", 
                        "Split Ratio", f"{100-test_size}% / {test_size}%")
        
        with tab4:
            st.markdown("### 📊 VISUALIZACIÓN AVANZADA")
            
            st.markdown("#### Sepal Length vs Petal Length por Especie")
            fig = px.scatter(df_scaled, 
                           x='sepal length (cm)', 
                           y='petal length (cm)',
                           color='species',
                           size='petal width (cm)',
                           hover_data=['sepal width (cm)'],
                           title="Análisis de Características Principales",
                           color_discrete_sequence=['#00f5ff', '#ff6b9d', '#ffd700'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0.1)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0', family='Rajdhani')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Matriz de Correlación")
            corr = df[iris.feature_names].corr()
            fig = px.imshow(corr, 
                           text_auto='.2f',
                           color_continuous_scale='RdBu_r',
                           title="Correlación entre Features")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Distribuciones por Pares de Features")
            fig = px.scatter_matrix(df,
                                   dimensions=iris.feature_names,
                                   color='species',
                                   title="Matriz de Dispersión Multivariable",
                                   color_discrete_sequence=['#00f5ff', '#ff6b9d', '#ffd700'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0.1)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0', size=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Distribución de Features por Especie")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(df, x='species', y='sepal length (cm)',
                           color='species',
                           title="Sepal Length por Especie",
                           color_discrete_sequence=['#00f5ff', '#ff6b9d', '#ffd700'])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0.1)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e0e0'),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, x='species', y='petal length (cm)',
                           color='species',
                           title="Petal Length por Especie",
                           color_discrete_sequence=['#00f5ff', '#ff6b9d', '#ffd700'])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0.1)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e0e0'),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")