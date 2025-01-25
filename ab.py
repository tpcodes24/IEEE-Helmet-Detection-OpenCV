## Connecting to the database

## importing 'mysql.connector' as mysql for convenient
import mysql.connector as mysql

## connecting to the database using 'connect()' method
## it takes 3 required parameters 'host', 'user', 'passwd'
db = mysql.connect(
    host = "localhost",
    user = "tejasree",
    passwd = "parasa",
    auth_plugin='mysql_native_password'
)

print(db) # it will print a connection object if everything is fine
