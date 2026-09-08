<div align="center">
  <h1><b>Clasificación supervisada de texto en redes sociales para la gestión de desastres naturales</b></h1>
  
  <h4>TRABAJO FIN DE GRADO</h4>
  <h4>Curso 2025/2026</h4>
  
  
  <br><br>
  
  Autora: Ana González Sánchez <br>

</div>

## Estructura del Repositorio
- Dataset/: Directorio que contiene los archivos train, test y dev de CrisisMMD (crisismmd_datasplit_all).
- resultados_TFG/: Todos los resultados de la experimentación
    - binario/: Métricas, umbrales óptimos ($u^*$), selecciones de modelos y predicciones para la tarea binaria.
    - multiclase/: Evaluaciones detalladas, búsqueda de pesos para el Ensemble ponderado y métricas por clase.
    - figuras/: Matrices de confusión en formato pdf y gráficas del preprocesado y del EDA
    - Curvas de aprendizaje/: Gráficos de curvas de aprendizaje.
    - logs/: Resúmenes de hiperparámetros y trazas de entrenamiento.
    - summary_metrics: Resumen de los resultados.
- Script del código fuente (tweet_classification.py)

## Requisitos e instalación
Para reproducir los experimentos y ejecutar el código de los modelos, es necesario disponer de un entorno de Python con las siguientes librerías principales:

`pip install pandas numpy matplotlib seaborn scikit-learn scipy torch transformers statsmodels`

(Nota: Se recomienda encarecidamente el uso de una GPU compatible con CUDA para acelerar el ajuste de hiperparámetros y el entrenamiento de los modelos basados en Transformers).

## Índice del código
El código fuente está segmentado en fases metodológicas bien diferenciadas:
- Fase 1: Carga de datos, análisis de concordancia de etiquetas y normalización sintáctica junto a la deduplicación de conjunto de datos.
- Fase 2: Análisis exploratorio de datos (EDA), estudio de la probabilidad condicional por desastre y cálculo de la V de Cramer.
- Fase 3: Modelización comparativa que abarca desde baselines clásicos (SVM con TF-IDF) y redes neuronales secuenciales (Bi-LSTM con embeddings GloVe congelados) hasta modelos de lenguaje preentrenados avanzados (BERT Base y BERTweet optimizados con Focal Loss).
- Fase 4: Inferencia sobre el conjunto de prueba, selección de umbrales óptimos maximizando $F_2$ en validación, evaluación de un Ensemble probabilístico y contraste estadístico mediante el test de McNemar.
