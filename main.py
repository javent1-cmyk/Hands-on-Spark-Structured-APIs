# main.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("MusicAnalysis").getOrCreate()

# Load datasets
logs = spark.read.csv("listening_logs.csv", header=True, inferSchema=True)
songs = spark.read.csv("songs_metadata.csv", header=True, inferSchema=True)


joined = logs.join(songs, on="song_id", how="inner")
print("JOINED COUNT:", joined.count())
joined = logs.join(songs, on="song_id", how="inner")

# Task 1: User Favorite Genres
genre_counts = joined.groupBy("user_id", "genre").agg(count("*").alias("listen_count"))

user_genre_window = Window.partitionBy("user_id").orderBy(desc("listen_count"))

favorite_genres = genre_counts.withColumn(
    "rank", row_number().over(user_genre_window)
).filter(col("rank") == 1).select(
    "user_id", "genre", "listen_count"
)

favorite_genres.write.mode("overwrite").csv(
    "outputs/task1_user_favorite_genres",
    header=True
)

# Task 2: Average Listen Time
avg_listen_time = logs.groupBy("user_id").agg(
    round(avg("duration_sec"), 2).alias("avg_duration_sec")
)

avg_listen_time.write.mode("overwrite").csv(
    "outputs/task2_average_listen_time",
    header=True
)

# Task 3: Genre Loyalty Scores - Top 10
total_listens = joined.groupBy("user_id").agg(
    count("*").alias("total_listens")
)

top_genres = favorite_genres.withColumnRenamed(
    "listen_count", "top_genre_listens"
).withColumnRenamed(
    "genre", "top_genree"
)

loyalty_scores = top_genres.join(total_listens, on="user_id").withColumn(
    "genre_loyalty_score",
    round((col("top_genre_listens") / col("total_listens")) * 100, 2)
).select(
    "user_id",
    col("top_genree").alias("top_genre"),
    "top_genre_listens",
    "total_listens",
    "genre_loyalty_score"
).orderBy(desc("genre_loyalty_score")).limit(10)

loyalty_scores.write.mode("overwrite").csv(
    "outputs/task3_genre_loyalty_top10",
    header=True
)

# Task 4: Night Owl Users 12 AM - 5 AM
night_owl_users = logs.withColumn(
    "timestamp_parsed", to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss")
).withColumn(
    "listen_hour", hour(col("timestamp_parsed"))
).filter(
    (col("listen_hour") >= 0) & (col("listen_hour") <= 5)
).select(
    "user_id"
).distinct()

night_owl_users.write.mode("overwrite").csv(
    "outputs/task4_night_owl_users",
    header=True
)

spark.stop()