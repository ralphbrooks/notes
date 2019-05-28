
---
title: "How to Set Environment Variables From A Linux Script"
author: "Ralph Brooks"
date: 2019-05-27T12:25:53-04:00
description: "This describes how to load environment variables from a script"
type: technical_note
draft: false
---

If you utilize the cloud, odds are that there has been a time where you spun up a machine and needed to set the environment 
variables from a linux script.

The process to do this is:

1) Create a file that contains environment variables that you want to export.

**environ.sh**
```bash
export APIKEY="Sample API key"
```

2) Change the permissions of the environment file:

```bash
chmod +x environ.sh
```

3) Execute the script with a dot space prefix. 

Use the dot space command as follows:

```. environ.sh```
