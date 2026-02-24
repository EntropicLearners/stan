Installation
============

We recommend installing Stan into a dedicated environment created using `uv` or `conda`.
You need to clone the Stan repository from Github and install it locally. 
It's not as difficult as it sounds, just follow these steps:

Open up the terminal and use ``cd`` to navigate to the directory where you want to store 
the Stan code. Clone the repo to your disk using:

.. code-block:: bash

   $ git clone https://github.com/EntropicLearners/stan

Since your terminal is currently in the Stan directory, you can now install Stan and 
all its dependencies with:

.. code-block:: bash

   $ pip install -e .

To keep your "local" Stan installation up to date, you should regularly pull 
the latest changes:

.. code-block:: bash

   $ git switch main
   $ git pull origin main

.. warning::
   For the development version of Stan to work, you need to first remove
   any versions that you may have previously installed using ``pip uninstall stan``