import hashlib

SECRET_KEY = 'django-insecure-hkrj5qe6)4-oe)g&+s-_)90r8$$fk_*a1w33=2wikt4!^4_h6c'

def md5(data_string):
    obj = hashlib.md5(SECRET_KEY.encode('utf-8'))
    obj.update(data_string.encode('utf-8'))
    return obj.hexdigest()

v = md5("123")
print(v)
