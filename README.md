# SerapisU33-AMF

The contents of this project are split into two parts:

## backend

Provides the backend services for the argument mining framework. These can be called independently, e.g.:

```
curl -XPOST -F “file=@data.txt” http://hostname/turninator-01
```

**Requirements:** Docker

## frontend

Provides a web-based GUI for selecting multiple AMF components and uploading a file to be processed by the selected pipeline.

**Config:** Edit the backend URL in am.php to match the install location of the backend components.

**Requirements:** Web server, PHP
