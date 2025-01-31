# Databricks notebook source
# MAGIC %md
# MAGIC # This is my titanic example

# COMMAND ----------

from pyspark.sql.functions import col, median, when, isnan, count, sum
from pyspark.sql.functions import mean, stddev
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql.types import IntegerType

# COMMAND ----------

df = spark.sql("SELECT * from titanic")
df.show(4)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fill in missing values

# COMMAND ----------

# Count null values in all columns
null_counts = df.select(
    *[sum(when(col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]
)

# Show the null counts
display(null_counts)

# COMMAND ----------

# Fill missing data in the age column with the mean age
median_age = df.select(median(col("Age"))).collect()[0][0]
df = df.withColumn("Age", when(col("Age").isNull(), median_age).otherwise(col("Age")))

# COMMAND ----------

# Count null values in the age column
null_count = df.select(
    count(
        when(col("Age").isNull() | isnan(col("Age")), "age")
    ).alias("null_count")
).collect()[0]["null_count"]

print(f"Number of null values in the age column: {null_count}")

# COMMAND ----------

from pyspark.sql.functions import col, when

# Calculate the mode of the embarked column
mode_embarked = df.groupBy("embarked").count().orderBy("count", ascending=False).first()[0]

# Replace null values in the embarked column with the mode
df = df.withColumn("embarked", when(col("embarked").isNull(), mode_embarked).otherwise(col("embarked")))

# Show the updated DataFrame
df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Remove column

# COMMAND ----------

# Drop the cabin column
df = df.drop("cabin")

# Show the updated DataFrame
df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Remove errors in data

# COMMAND ----------

# Remove errors in data (example: negative ages)
df = df.filter(col("Age") >= 0)

# COMMAND ----------

negative_age_count = df.filter(col("age") < 0).count()
print(f"Number of negative values in the age column: {negative_age_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Remove outliers

# COMMAND ----------

df.count()

# COMMAND ----------

# Remove outliers (example: age > 3 standard deviations from the mean)
age_mean = df.select(mean(col("Age"))).collect()[0][0]
age_stddev = df.select(stddev(col("Age"))).collect()[0][0]
df = df.filter(
    (col("Age") >= age_mean - 3 * age_stddev) & 
    (col("Age") <= age_mean + 3 * age_stddev)
)

display(df)

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fix data type

# COMMAND ----------

# Change data type (example: PassengerID to Integer)
df = df.withColumn("PassengerID", col("PassengerID").cast(IntegerType()))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fix unbalanced data

# COMMAND ----------

# Fix unbalanced data (example: oversample the minority class)
major_df = df.filter(col("Survived") == 0)
minor_df = df.filter(col("Survived") == 1)
ratio = major_df.count() / minor_df.count()
minor_df = minor_df.sample(withReplacement=True, fraction=ratio)
df = major_df.union(minor_df)

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Derive new features

# COMMAND ----------

# Create a new feature (example: family_size = sibsp + parch + 1)
df = df.withColumn("family_size", col("sibsp") + col("parch") + 1)

# COMMAND ----------

df.show(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Normalize numeric columns

# COMMAND ----------

# Normalize numeric columns
numeric_cols = ["Age", "Fare", "family_size"]
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
pipeline = Pipeline(stages=[assembler, scaler])
df = pipeline.fit(df).transform(df)

# COMMAND ----------

display(df.limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### One Hot Encoding

# COMMAND ----------

# Apply one hot encoding
categorical_cols = ["Sex", "Embarked", "Pclass"]
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_ohe") for col in categorical_cols]
pipeline = Pipeline(stages=indexers + encoders)
df = pipeline.fit(df).transform(df)

# COMMAND ----------

display(df.limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training and evaluating model

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

# Split the data into training and test sets (80% training, 20% test)
train_df, test_df = df.randomSplit([0.8, 0.2], seed=123)

# COMMAND ----------

# Assemble the features into a single vector
feature_cols = ["scaled_features", "Embarked_ohe", "Pclass_ohe"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="output_features")

# COMMAND ----------

# Initialize the logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="Survived")

# COMMAND ----------

# Create a pipeline
pipeline = Pipeline(stages=[assembler, lr])

# COMMAND ----------

# Train the model
model = pipeline.fit(train_df)

# COMMAND ----------

# Make predictions on the test set
predictions = model.transform(test_df)

# COMMAND ----------

# Show the predictions
predictions.select("PassengerID", "survived", "prediction", "probability").show()

# COMMAND ----------

# Evaluate the model.Default
evaluator = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderROC")
roc = evaluator.evaluate(predictions)
print(f"Test set area under ROC: {roc}")

# COMMAND ----------

# Initialize the evaluator with the area under precision-recall curve
evaluator = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderPR")

# Evaluate the model on the test set
areaUnderPR = evaluator.evaluate(predictions)

print(f"Test set area under PR score: {areaUnderPR}")

# COMMAND ----------


