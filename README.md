# howtotreatscars

## Setting up environment using pip and virtualenv
Assuming python 3.11 (might work with lower versions but not sure)
```bash
pip install virtualenv # install virtualenv package if you do not have it already
virtualenv venv # create locally a new virtualenv called venv
source venv/bin/activate # activate the venv
pip install -r requirements.txt # install required packages within the venv
```


## launching streamlit app
`python3 -m streamlit run interface.py`

## setting the password
Create a file `.streamlit/secrets.toml`, and fill it with
```bash
password = "mypassword"
openai_api_key = "mykey"
```