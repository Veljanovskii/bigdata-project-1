# Importing packages
import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from dotenv import load_dotenv
from datetime import datetime
from geopy.distance import geodesic
from pyspark.sql.types import FloatType


@udf(returnType=FloatType())
def geodesic_udf(a, b):
    return geodesic(a, b).m


if __name__ == "__main__":

    spark = SparkSession.builder.appName('PySparkApp').getOrCreate()
    startTime = datetime.now()

    # Loading arguments

    load_dotenv()
    SAN_JOSE_LATITUDE = float(os.getenv('SAN_JOSE_LATITUDE'))
    SAN_JOSE_LONGITUDE = float(os.getenv('SAN_JOSE_LONGITUDE'))
    SAN_FRANCISCO_LATITUDE = float(os.getenv('SAN_FRANCISCO_LATITUDE'))
    SAN_FRANCISCO_LONGITUDE = float(os.getenv('SAN_FRANCISCO_LONGITUDE'))
    YEAR_START = int(os.getenv('YEAR_START'))
    MONTH_START = int(os.getenv('MONTH_START'))
    DAY_START = int(os.getenv('DAY_START'))
    YEAR_END = int(os.getenv('YEAR_END'))
    MONTH_END = int(os.getenv('MONTH_END'))
    DAY_END = int(os.getenv('DAY_END'))
    FILE_DATA = os.getenv('FILE_DATA')
    HDFS_DATA = os.getenv('HDFS_DATA')

    # Reading csv file
    df = spark.read.csv('./baywheels/baywheels.csv', header=True)
    #df = spark.read.csv(FILE_DATA, header=True)
    #df = spark.read.csv(HDFS_DATA, header=True)

    output = list()

    # Add ride duration column
    timeDiff = (unix_timestamp('ended_at', 'yyyy-MM-dd HH:mm:ss') - unix_timestamp('started_at', 'yyyy-MM-dd HH:mm:ss'))
    df = df.withColumn('duration', timeDiff)

    # Add distance column
    df = df.withColumn('distance', geodesic_udf(array('start_lat', 'start_lng'), array('end_lat', 'end_lng')))

    # Add average speed column
    df = df.withColumn('average_speed', col('distance')/col('duration'))

    # Fill 0 where null
    df = df.na.fill(0)

    # Task 1
    task1 = df.select(df.start_station_name, df.end_station_name).filter((df.start_lat < SAN_JOSE_LATITUDE) &
        (df.start_lng > SAN_JOSE_LONGITUDE) & (df.end_lat > SAN_FRANCISCO_LATITUDE) & (df.end_lng < SAN_FRANCISCO_LONGITUDE)).collect()
    output.append('Task 1:\n\n''Trips from San Jose to San Francisco:\n')
    print('\nTrips from San Jose to San Francisco:\n')
    for row in task1:
        print(str(row['start_station_name']) + ", " + str(row['end_station_name']))
        output.append(str(row['start_station_name']) + ", " + str(row['end_station_name']))

    # Task 2
    startDate = datetime(YEAR_START, MONTH_START, DAY_START)
    endDate = datetime(YEAR_END, MONTH_END, DAY_END)
    task2 = df.groupBy('rideable_type').agg(count(when((col('started_at') > startDate) & (col('ended_at') < endDate), True))
        .alias('number_of_rides')).collect()
    task2_result = 'Number of bike rides in the given period: for \'' + str(task2[0].asDict()['rideable_type']) + '\' is ' \
        + str(task2[0].asDict()['number_of_rides']) + ', for \'' + str(task2[1].asDict()['rideable_type']) + '\' is ' \
        + str(task2[1].asDict()['number_of_rides']) + ', and for \'' + str(task2[2].asDict()['rideable_type']) + '\' is ' \
        + str(task2[2].asDict()['number_of_rides'])
    print(task2_result, '\n')
    output.append('\nTask 2:\n')
    output.append(task2_result + '\n')

    # Task 3
    task3 = df.groupBy('member_casual').agg(avg('duration').alias('average_trip_time')).filter(col('average_trip_time') > 300).collect()
    task3_result = str('Average trip duration time for ' + task3[0].asDict()['member_casual']) + ' riders: ' \
        + str(task3[0].asDict()['average_trip_time'] / 60) + ' minutes. Average trip duration time for ' \
        + str(task3[1].asDict()['member_casual'] + 's: ' + str(task3[1].asDict()['average_trip_time'] / 60) + ' minutes.')
    print(task3_result, '\n')
    output.append('Task 3:\n')
    output.append(task3_result + '\n')

    # Task 4
    task4 = df.groupBy('end_station_name').agg(count('ride_id').alias('ride_count'), mean('duration').alias('mean_trip_time'))\
        .filter(col('end_station_name').like('%Van Ness Ave%'))\
        .sort(col('mean_trip_time').asc()).collect()
    output.append('Task 4:\n')
    for row in task4:
        task4_result = 'Number of bike rides that ended on the \'' + str(row.asDict()['end_station_name']) + '\' is ' \
            + str(row.asDict()['ride_count']) + ' with mean time of ' + str(row.asDict()['mean_trip_time']) + ' seconds.'
        print(task4_result)
        output.append(task4_result)

    # Task 5
    task5 = df.groupBy('start_station_name', 'end_station_name').agg({'duration': 'stddev'}) \
        .withColumnRenamed('stddev(duration)', 'standard_deviation_time').sort(
        col("standard_deviation_time").desc()).collect()
    task5_result = 'The highest standard deviation for the trip duration is for the route between ' \
                   + str(task5[0].asDict()['start_station_name']) + ' and ' \
                   + str(task5[0].asDict()['end_station_name']) + '.'
    print(task5_result, '\n')
    output.append('\nTask 5:\n')
    output.append(task5_result + '\n')

    endTime = datetime.now()
    executionTime = endTime - startTime

    print('Local spark execution was ' + str(executionTime) + ' long.')
    output.append('\nLocal spark execution was ' + str(executionTime) + ' long.')

    with open('output.txt', 'w') as fileOutput:
        for line in output:
            fileOutput.write(line)
            fileOutput.write('\n')

    spark.stop()
