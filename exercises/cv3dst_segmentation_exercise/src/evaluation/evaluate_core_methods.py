import os
import sys
import traceback
import mysql
import mysql.connector
from mysql.connector import Error
from mysql.connector import errorcode
import csv

motchallenge_metric_names = {
    "idf1": "IDF1",
    "recall": "Rcll",
    "precision": "Prcn",
    "num_unique_objects": "GT",
    "mostly_tracked": "MT",
    "mostly_lost": "ML",
    "num_false_positives": "FP",
    "num_misses": "FN",
    "num_switches": "IDSW",
    "mota": "MOTA",
}


def write_result_db(connection, user, summary, subs, bonus, mode="live"):
    if mode == "live":
        db_name = "cv3dst"
    elif mode == "dev":
        db_name = "cv3dst_dev"
    sequences = summary.index
    # write sql update query
    for seq in sequences:
        update_query_metrics = "INSERT INTO `{}`.`Leaderboard` (`login`, `sequence`, `submission_nr`, `pass`,".format(
            db_name
        )
        update_values = "VALUES('{}', '{}', {}, {}, ".format(user, seq, subs + 1, bonus)
        for key, value in motchallenge_metric_names.items():
            update_values += "'{}',".format(summary.loc[seq, key])
            update_query_metrics += "`{}`,".format(value)
        update_values = update_values[:-1] + ")"
        update_query_metrics = update_query_metrics[:-1] + ") "
        update_query = update_query_metrics + update_values
        cursor = connection.cursor()
        cursor.execute(update_query)
        connection.commit()


def check_submission(connection, login, mode="live"):
    if mode == "live":
        db_name = "cv3dst"
    elif mode == "dev":
        db_name = "cv3dst_dev"

    query_submission = "SELECT max(`submission_nr`) as max_sub FROM `{}`.`Leaderboard` where login='{}'".format(
        db_name, login
    )
    cursor = connection.cursor()
    result = cursor.execute(query_submission)
    row = cursor.fetchone()
    subs = row[0]
    if subs == None:
        subs = 0

    return subs


def establish_conn(mode="live"):
    connection = None
    db_user = "cv3dst"
    password = "geep4aiC5teikoo9"
    host = "sql9.in.tum.de"
    try:
        if mode == "live":
            db_name = "cv3dst"
            connection = mysql.connector.connect(host=host, database=db_name, user=db_user, password=password)
        elif mode == "dev":
            db_name = "cv3dst_dev"
            connection = mysql.connector.connect(host=host, database=db_name, user=db_user, password=password)
        else:
            raise ValueError("The execution mode you have specified is not valid.")
            print(connection)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("<br> Something is wrong with your user name or password <br>")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("<br> Database does not exist <br>")
        else:
            print(err)
    return connection


def establish_conn_test(mode="live"):
    connection = None
    db_user = "cv3dst"
    password = "geep4aiC5teikoo9"
    host = "sql9.in.tum.de"
    try:
        if mode == "live":
            db_name = "cv3dst"
            connection = mysql.connector.connect(host=host, database=db_name, user=db_user, password=password)
        elif mode == "dev":
            db_name = "cv3dst_dev"
            connection = mysql.connector.connect(host=host, database=db_name, user=db_user, password=password)
        else:
            raise ValueError("The execution mode you have specified is not valid.")
            print(connection)
        db_name = "cv3dst_ex3_dev"
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("<br> Something is wrong with your user name or password <br>")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("<br> Database does not exist <br>")
        else:
            print(err)
    return connection


def write_score_to_database(login, sub_scores, mode):
    """
    Establishes a database connection to post evaluated score.
    :param login: login name, e.g. s172
    :param exercise: exercise number in database
    :param score: total achieved score to written to database for this exercise. Includes [score, pass, submission_nr]
    """
    if mode == "live":
        db = MySQLdb.connect("sqlradig.informatik.tu-muenchen.de", "dl4cv", "hjaDgX9mTC41djDG", "dl4cv")
    elif mode == "dev":
        db = MySQLdb.connect("sqlradig.informatik.tu-muenchen.de", "dl4cv", "hjaDgX9mTC41djDG", "dl4cv_dev")
    else:
        raise ValueError("The execution mode you have specified is not valid.")

    cursor = db.cursor()

    try:

        score, bonus, submission_nr = sub_scores

        # fill data base for scores and bonus pass

        query_current_score = "SELECT score FROM Leaderboard WHERE login='{0}' AND exercise={1}".format(
            login, submission_nr
        )
        cursor.execute(query_current_score)
        result = cursor.fetchone()

        if result == None or result == (None,) or score > result[0]:
            sql = "INSERT INTO Leaderboard(login, exercise, score) VALUES ('{0}', {1}, {2}) ON DUPLICATE KEY UPDATE score={2}".format(
                login, submission_nr, score
            )

            cursor.execute(sql)

            query_current_score = "SELECT pass_1 FROM Leaderboard WHERE login='{0}' AND exercise={1}".format(
                login, submission_nr
            )

            cursor.execute(query_current_score)
            result = cursor.fetchone()

            sql = "INSERT INTO Leaderboard(login, exercise, pass_1) VALUES ('{0}', {1}, {2}) ON DUPLICATE KEY UPDATE pass_1={2}".format(
                login, submission_nr, bonus
            )

            cursor.execute(sql)

        db.commit()
    except Exception as e:
        db.rollback()
        print(("Rolled back. Error: " + str(e)))

    db.close()


def evaluation_main(evaluation_function, submission_nr, model_path, threshold, mode="live"):
    try:
        login = os.environ.get("USER")
        if login == None:
            raise ValueError("You did not provide a valid login environment variable.")

        # load csv file and check for developers
        with open("dev_accounts_ws19.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                if row[0] == login:
                    mode = "dev"

        score = evaluation_function(model_path)
        if score > threshold:
            passed = 1
            print("This model reached the required score in order to be eligible for the bonus points!")
        else:
            passed = 0
            print("This model did not reach the required score in order to be eligible for the bonus points!")

        print(("The achieved score for this model is: %02.2f\n" % score))
        write_score_to_database(login, (score, passed, submission_nr), mode)

        return passed

    except:
        print("The evaluation of the submission failed with:\n")
        print(traceback.format_exc(limit=-1))
        exit(1)
