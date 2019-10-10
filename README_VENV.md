The BlueSky dependency pyopengl-accelerate fails to install on Python 3.7 (but you don't *need* it).
Below are instructions for installing any Python 3.x version non-intrusively on MacOS.

- Install HomeBrew
- `brew install zlib`
- `brew install tcl-tk`

To configure tcl-tk and zlib for use by pyenv,
- Add the following lines to your *.bashrc* or *.zshrc* file,<br>
`export LDFLAGS="-L/usr/local/opt/tcl-tk/lib"`<br>
`export CPPFLAGS="-I/usr/local/opt/tcl-tk/include"`<br>
`export PATH=$PATH:/usr/local/opt/tcl-tk/bin`<br>
`export LDFLAGS="${LDFLAGS} -L/usr/local/opt/zlib/lib"`<br>
`export CPPFLAGS="${CPPFLAGS} -I/usr/local/opt/zlib/include"`<br>
`export PKG_CONFIG_PATH="${PKG_CONFIG_PATH} /usr/local/opt/zlib/lib/pkgconfig"`
- Restart your shell

To set up a virtual environment within a pyenv environment,
- `brew install pyenv`
- `pyenv install 3.6.4`
- `brew install pyenv-virtualenv`
- Add the following lines to your *.bash_profile* (or *.zshrc*) file :<br>
`eval "$(pyenv init -)"`<br>
`eval "$(pyenv virtualenv-init -)"`
- Restart your shell
- `pyenv virtualenv 3.6.4 venv_BSky`
- `pyenv activate venv_BSky`
- Verify using `python -V`. 
- `python -m pip install -r requirements.txt`, `python -m install pyopengl-accelerate`

I ran into tkinter-related import error while running BlueSky in a venv, so if you're able to get it running, let me know!