from typing import List

from gensim.topic_coherence.indirect_confirmation_measure import cosine_similarity
from implicit.als import AlternatingLeastSquares
import numpy as np
import scipy.sparse as sp
from fastapi import FastAPI
from pydantic import BaseModel
from gensim.models import Word2Vec

# python -m uvicorn app:app --host 0.0.0.0 --port 8000

app = FastAPI()

# глобальные переменные
ALSModel = None  # модель для коллаборативной фильтрации
user_item_matrix = None  # матрица взаимодействий

Word2VecModel = None # модель для контентно-ориентированной фильтрации
dish_vectors = None # вектора всех блюд


class FitCollaborativeFilteringRequest(BaseModel):
    userItemMatrix: List[List[int]]


class FitContentBasedRequest(BaseModel):
    dishesIngredients: List[List[str]]


class RecommendRequest(BaseModel):
    userId: int


@app.post("/fit-collaborative-filtering-model")
async def fit_model(data: FitCollaborativeFilteringRequest):
    """Обучение модели ALS на матрице взаимодействий пользователь-блюдо"""
    global ALSModel, user_item_matrix

    print("POST /fit-collaborative-filtering-model")

    # получаем разреженную матрицу
    user_item_matrix = sp.csr_matrix(data.userItemMatrix)

    # обучаем модель
    ALSModel = AlternatingLeastSquares(factors=20, iterations=40)
    ALSModel.fit(user_item_matrix)

    return {"message": "ALS model trained successfully"}


@app.post("/fit-content-based-model")
async def fit_model(data: FitContentBasedRequest):
    """Обучение модели Word2Vec на блюдах в виде списка ингредиентов"""
    global Word2VecModel, dish_vectors

    print("POST /fit-content-based-model")

    # обучаем модель
    Word2VecModel = Word2Vec(sentences=data.dishesIngredients, vector_size=100, window=5, min_count=1, workers=4)

    # превращаем блюда в вектора
    dish_vectors = [get_dish_vector(dish) for dish in data.dishesIngredients]

    return {"message": "Word2Vec model trained successfully"}


@app.post("/recommend")
async def recommend(data: RecommendRequest):
    """Получение рекомендаций для пользователя"""
    global ALSModel, user_item_matrix

    print("POST /recommend")

    if ALSModel is None or user_item_matrix is None:
        return {"error": "Model is not trained yet. Train the model first using POST /fit-collaborative-filtering-model"}

    user_id = data.userId
    recommendationsFromCollaborative = collaborativeFiltering(user_id)
    recommendationsFromContentBased = contentBased(user_id)

    recommendations = recommendationsFromCollaborative объединить с recommendationsFromContentBased
    return {"recommendations": recommendations}


def collaborativeFiltering(user_id):
    global ALSModel, user_item_matrix

    if ALSModel is None or user_item_matrix is None:
        return {"error": "ALS model is not trained yet. Train the model first using POST /fit-collaborative-filtering-model"}

    item_ids, scores = ALSModel.recommend(
        userid=user_id,
        user_items=user_item_matrix[user_id]
    )

    recommendations = [{"item_id": int(item_id), "score": float(score)}
                       for item_id, score in zip(item_ids, scores)]

    return {"recommendations": recommendations}


def contentBased(user_id):
    global user_item_matrix, dish_vectors

    if user_item_matrix is None or dish_vectors is None:
        return {"error": "Content-based model is not trained yet. Train the model first using POST /fit-content-based-model"}

    # Получаем список блюд, которые пользователь уже выбрал
    user_interactions = user_item_matrix[user_id].toarray().flatten()
    liked_dishes = np.where(user_interactions > 0)[0]

    if len(liked_dishes) == 0:
        return []

    # Рассчитываем сходство каждого блюда с понравившимися пользователю
    user_dish_vectors = [dish_vectors[dish_id] for dish_id in liked_dishes]
    similarities = cosine_similarity(user_dish_vectors, dish_vectors)

    # Среднее сходство по всем понравившимся блюдам
    mean_similarity = np.mean(similarities, axis=0)

    # Получаем топ-N блюд
    top_indices = mean_similarity.argsort()[::-1][:10]  # 10 лучших рекомендаций
    recommendations = [{"item_id": int(index), "score": float(mean_similarity[index])} for index in top_indices]

    return {"recommendations": recommendations}


def get_dish_vector(dish):
    global Word2VecModel

    if ALSModel is None or user_item_matrix is None:
        return {"error": "Word2Vec model is not trained yet. Train the model first using POST /fit-content-based-model"}

    # Фильтруем ингредиенты, которые есть в словаре модели
    valid_ingredients = [Word2VecModel.wv[ingredient] for ingredient in dish if ingredient in Word2VecModel.wv]

    # Возвращаем средний вектор блюда
    return np.mean(valid_ingredients, axis=0) if valid_ingredients else np.zeros(Word2VecModel.vector_size)
