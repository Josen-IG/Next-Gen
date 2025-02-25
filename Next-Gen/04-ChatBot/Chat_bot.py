import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from fuzzywuzzy import process
import groq  # Importamos la API de Groq para IA

# 📌 **Configurar la API de Groq**
GROQ_API_KEY = "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"  # Reemplázalo con tu clave
client = groq.Client(api_key=GROQ_API_KEY)

# 📌 **Función para normalizar nombres de jugadores con Groq**
def normalizar_nombre_groq(nombre_usuario):
    prompt = f"""
    Eres un experto en fútbol con conocimientos actualizados sobre jugadores de todas las ligas.
    
    Tarea:
    - Si "{nombre_usuario}" es un apodo, abreviación o variación del nombre de un jugador de fútbol, conviértelo a su nombre completo y correcto.
    - Si ya está bien escrito, devuélvelo igual.
    - Si el nombre no pertenece a ningún futbolista conocido, responde solo con "Desconocido" (sin comillas).
    
    Ejemplos:
    - "Vini jr" → "Vinícius José de Oliveira Júnior"
    - "CR7" → "Cristiano Ronaldo"
    - "Leo" → "Lionel Messi"
    - "K Mbappe" → "Kylian Mbappé"
    - "Pedri" → "Pedro González López"
    - "Pelé" → "Edson Arantes do Nascimento"
    
    Responde solo con el nombre corregido. No agregues explicaciones ni texto adicional. Se directo y fromal con tu respuesta  **RESPONDE UNICA Y EXCLUSIVAMENTE EN ESPAÑOL**
    """
    response = client.chat.completions.create(
        model="llama3-8b-8192", 
        messages=[{"role": "system", "content": "Eres un experto en fútbol y SIEMPRE responde en español."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# 📌 **Función para encontrar el nombre más parecido en la base de datos**
def encontrar_nombre_similar(nombre_usuario, lista_nombres):
    mejor_match = process.extractOne(nombre_usuario, lista_nombres)
    return mejor_match[0] if mejor_match[1] > 85 else None  # Umbral de similitud aumentado

# 📌 **Función para cargar el dataset**
@st.cache_data
def cargar_datos():
    df = pd.read_csv("skills_resultado.csv")  # Ajusta la ruta si es necesario
    return df

df = cargar_datos()
jugadores = df["name"].unique()

# 📌 **Comparador de Jugadores con Radar Chart**
st.title("⚽ Comparador de Jugadores")

# **Entrada de jugadores por el usuario**
jugador1_input = st.text_input("🔹 Escribe el nombre del primer jugador:")
jugador2_input = st.text_input("🔸 Escribe el nombre del segundo jugador:")

if jugador1_input and jugador2_input:
    # 🔹 **Normalizar nombres con Groq**
    jugador1 = normalizar_nombre_groq(jugador1_input)
    jugador2 = normalizar_nombre_groq(jugador2_input)

    # 🔹 **Buscar los nombres más parecidos en la base de datos**
    jugador1_match = encontrar_nombre_similar(jugador1, jugadores)
    jugador2_match = encontrar_nombre_similar(jugador2, jugadores)

    if jugador1_match and jugador2_match:
        st.write(f"✅ **Jugadores seleccionados:**\n - {jugador1_input} → {jugador1_match}\n - {jugador2_input} → {jugador2_match}")

        # **Filtrar los datos de los jugadores seleccionados**
        df_selected = df[df["name"].isin([jugador1_match, jugador2_match])]

        # **Atributos a comparar**
        attributes = ["pace_total", "shooting_total", "passing_total", 
                      "dribbling_total", "defending_total", "physicality_total"]
        attributes_labels = ["Velocidad", "Tiro", "Pase", "Regate", "Defensa", "Físico"]

        # **Extraer valores de los jugadores**
        values1 = df_selected[df_selected["name"] == jugador1_match][attributes].values.flatten()
        values2 = df_selected[df_selected["name"] == jugador2_match][attributes].values.flatten()

        # **Crear Radar Chart**
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
        fig.patch.set_alpha(0)  
        ax.set_facecolor("none")  

        # **Crear ángulos para cada variable**
        angles = np.linspace(0, 2 * np.pi, len(attributes), endpoint=False).tolist()
        values1 = np.concatenate((values1, [values1[0]]))
        values2 = np.concatenate((values2, [values2[0]]))
        angles += angles[:1]

        # **Dibujar las líneas del Radar Chart**
        ax.plot(angles, values1, color="deepskyblue", linewidth=3, linestyle="-", label=jugador1_match)
        ax.plot(angles, values2, color="darkorange", linewidth=3, linestyle="-", label=jugador2_match)

        # **Configurar etiquetas del radar**
        new_order = [1, 0, 3, 2, 4, 5]  # Nuevo orden: Tiro, Velocidad, Regate, Pase, Defensa, Físico
        new_attributes_labels = [attributes_labels[i] for i in new_order]
        ax.set_xticks(angles[:-1])  
        ax.set_xticklabels(new_attributes_labels, fontsize=14, color="white", fontweight="bold")  
        ax.set_yticklabels([])  
        ax.spines["polar"].set_color("white")  
        ax.grid(color="gray", linestyle="--", linewidth=0.8)  

        # **Aumentar la distancia de las etiquetas del centro**
        for label, angle in zip(ax.get_xticklabels(), angles):
            label.set_horizontalalignment('center')
            label.set_verticalalignment('center')
            x = label.get_position()[0]
            y = label.get_position()[1]
            label.set_x(x * 5)  # Ajusta el factor de multiplicación según sea necesario
            label.set_y(y * 5)  # Ajusta el factor de multiplicación según sea necesario

        # **Configurar la leyenda**
        legend = ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), facecolor="none")
        for text in legend.get_texts():
            text.set_color("white")

        # **Mostrar gráfico en Streamlit**
        st.pyplot(fig)

        # 📌 **Obtener justificación de similitudes con Groq**
        def justificar_similitudes(jugador1, jugador2, df):
            prompt = f"""
            Actúa como un analista de fútbol. Compara a los jugadores {jugador1} y {jugador2} en base a los siguientes atributos:
            {attributes_labels}.
            Explica en qué aspectos son similares y en qué aspectos difieren.
            """
            response = client.chat.completions.create(
                model="llama3-8b-8192", 
                messages=[{"role": "system", "content": "Eres un experto en fútbol."},
                          {"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()

        justificacion = justificar_similitudes(jugador1_match, jugador2_match, df_selected)
        st.subheader("📊 Justificación de similitudes:")
        st.write(justificacion)

        # 📌 **Lista de los 5 jugadores más similares al jugador 1**
        def encontrar_5_mas_parecidos(jugador, df):
            df["similitud"] = df["name"].apply(lambda x: process.extractOne(jugador, [x])[1])
            df_similares = df.sort_values(by="similitud", ascending=False).head(6)  
            return df_similares.iloc[1:6]["name"].tolist()  

        top_5_similares = encontrar_5_mas_parecidos(jugador1_match, df)
        st.subheader(f"🔍 Top 5 jugadores más parecidos a {jugador1_match}:")
        st.write("\n".join([f"- {player}" for player in top_5_similares]))

    else:
        st.error("❌ No se encontraron coincidencias para uno o ambos jugadores. Intenta nuevamente.")