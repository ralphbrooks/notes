
---
title: "Setting Environment Variables From Script"
author: "Ralph Brooks"
date: 5/19/2019
description: "This describes how to load environment variables from a script"
type: technical_note
draft: false
---

If you are spinning up different machines, it may be helpful to have environment variables set via script.

The file might look like

**environ.sh**
```bash
export APIKEY="Sample API key"
```

To use this on the spun-up machine, use the dot space command as follows:

```. environ.sh```