# Gold-Recovery-Prediction-Via-Regression

![alt image](https://github.com/AeroGenCreator/Gold-Recovery-Prediction-Via-Regression/blob/main/images/cover.jpeg)

## Índice del Flujo de Trabajo [Estructura del Notebook](https://github.com/AeroGenCreator/Gold-Recovery-Prediction-Via-Regression/blob/main/Proyecto%20Sprint%2013.ipynb)

1. Carga y Preparación de Datos:

        Importación de librerías, carga de datasets (train, test, full) y tratamiento de valores nulos.

3. Introducción y Contexto:

        Descripción del proceso de extracción de oro.

5. Análisis Exploratorio de Datos (EDA):

        Validación del cálculo de recuperación (Recovery).
    
        Análisis de la concentración de metales (Au, Ag, Pb) por etapa.
    
        Comparación de distribuciones de partículas entre entrenamiento y prueba.
    
        Estudio de valores atípicos (outliers).

6. Preprocesamiento para Machine Learning:

        Sincronización de características entre conjuntos.
    
        Segmentación del proceso en dos etapas: Rougher y Final.
    
        Escalado de variables mediante StandardScaler.

7. Desarrollo del Modelo:
    
        Implementación de la métrica personalizada sMAPE.
    
        Configuración de make_scorer para validación cruzada.
    
        Entrenamiento y evaluación multimodelo (Ridge, Random Forest, Gradient Boosting) usando K-Fold Cross-Validation.

8. Evaluación Final y Resultados:
    
       Predicciones en el conjunto de prueba y cálculo del sMAPE ponderado final.

10. Conclusiones y Exportación:
   
        Resumen de hallazgos y persistencia de modelos/objetos.

## Optimización de la Recuperación de Oro mediante Machine Learning

![alt image](https://github.com/AeroGenCreator/Gold-Recovery-Prediction-Via-Regression/blob/main/images/snap_1.png)

## [DASHBOARD](https://gold-recovery-prediction-via-regression.onrender.com)

📝 Descripción del Proyecto

Este proyecto simula el proceso tecnológico de extracción de oro de la minería real. El objetivo es predecir la cantidad de oro recuperado del mineral de oro mediante modelos de regresión, optimizando la eficiencia de la planta de producción y ayudando a descartar parámetros desfavorables.
📊 Puntos Clave del Análisis (EDA)

    Dinámica de los Metales: Se visualizó cómo la concentración de Oro (Au) aumenta linealmente conforme avanza el 
    proceso (Rougher -> Primary Cleaner -> Final), mientras que otros metales como la Plata (Ag) disminuyen.

    Consistencia de Datos: Se realizó un análisis de distribución de partículas para asegurar que el conjunto de 
    entrenamiento y prueba fueran estadísticamente comparables, garantizando la fiabilidad del modelo.

    Tratamiento de Datos Reales: Limpieza de valores ausentes basados en la continuidad del proceso tecnológico.

⚙️ Implementación Técnica Relevante
1. Métrica Personalizada: sMAPE

Para este proyecto, se implementó el Error Medio Absoluto Porcentual Simétrico (sMAPE). A diferencia del MAE convencional, el sMAPE es ideal para comparar errores en diferentes escalas de valores.

$$ sMAPE = \frac {1}{N} \sum_{i=1}^N \frac {|y - \hat{y}_i|}{(|y|+|\hat{y}_i|)}$$

Se integró en el ecosistema de Scikit-Learn utilizando `make_scorer`, permitiendo su uso directo en funciones de optimización.
2. Evaluación Multimodelo con Cross-Validation

No nos conformamos con un solo algoritmo. Implementamos una estrategia de K-Fold Cross-Validation (6 splits) para evaluar:

    Ridge Regression

    Random Forest Regressor

    Gradient Boosting Regressor

🏆 Resultados Finales

El modelo final se construyó bajo un esquema de dos etapas (Rougher y Final), logrando un desempeño excepcional:

    sMAPE Etapa Rougher: 0.72%

    sMAPE Etapa Final: 1.44%

    sMAPE Ponderado Final: 1.26% 🚀

Este bajo error porcentual demuestra la robustez de los modelos (especialmente Gradient Boosting) para predecir la recuperación con alta precisión.

🛠️ Tecnologías Utilizadas

    Python: Pandas, NumPy, Scipy.

    Visualización: Seaborn, Matplotlib.

    Machine Learning: Scikit-Learn (StandardScaler, Cross-validation, GradientBoosting).

    Model Deployment Ready: Exportación de modelos y escaladores mediante joblib.
