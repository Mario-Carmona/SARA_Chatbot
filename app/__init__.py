#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask


app = Flask(__name__)

import views

if __name__ == "__main__":
    app.run()