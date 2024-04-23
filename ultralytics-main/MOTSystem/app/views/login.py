from django.shortcuts import render, HttpResponse, redirect
from django.core.validators import RegexValidator
from django import forms
from io import BytesIO

from app.utils.code import check_code
from app import models
from app.utils.bootstrap import BootStrapForm
from app.utils.encrypt import md5

class LoginForm(BootStrapForm):
    username = forms.CharField(
        min_length=4,
        label="用户名",
        widget=forms.TextInput(),
        required=True,

    )
    password = forms.CharField(
        min_length=6,
        label="密码",
        widget=forms.PasswordInput(render_value=True),
        required=True,
    )
    email = forms.EmailField(
        label="邮箱",
        widget=forms.TextInput,
        required=True
    )
    qq = forms.CharField(
        min_length=5,
        max_length=12,
        label="QQ",
        widget=forms.TextInput,
        required=True,
        validators = [RegexValidator(r'^[1-9][0-9]*$', '只能为数据，并大于5位小于12位'), ],
    )
    code = forms.CharField(
        label="验证码",
        widget=forms.TextInput,
        required=True
    )
    def __init__(self, *args, **kwargs):
        condition = kwargs.pop('condition', False)
        super(LoginForm, self).__init__(*args, **kwargs)

        if condition == 'login':
            self.fields['username'].required = True
            self.fields['password'].required = True
            self.fields['email'].required = False
            self.fields['qq'].required = False
            # self.fields['username'].widget = forms.TextInput(attrs={'class': 'form-group hidden'})
        elif condition == 'signup':
            self.fields['username'].required = True
            self.fields['password'].required = True
            self.fields['email'].required = True
            self.fields['qq'].required = True
        elif condition == 'forget':
            self.fields['username'].required = False
            self.fields['password'].required = False
            self.fields['email'].required = True
            self.fields['qq'].required = False

    def clean_password(self):
        pwd = self.cleaned_data.get("password")
        return md5(pwd)

loginCss = {'username': '',
            'password': '',
            'email': 'hidden',
            'qq': 'hidden',
            'login': 'active',
            'signup': '',
            'forget': '',
            'table_id': 'login'}
signupCss = {'username': '',
            'password': '',
            'email': '',
            'qq': '',
            'login': '',
            'signup': 'active',
            'forget': '',
            'table_id': 'signup'}
forgetCss = {'username': 'hidden',
            'password': 'hidden',
            'email': '',
            'qq': 'hidden',
            'login': '',
            'signup': '',
            'forget': 'active',
            'table_id': 'forget'}
def login(request):
    """ 登录 """
    if request.method == "GET":
        form = LoginForm()
        return render(request, 'login.html', {'form': form, 'css': loginCss})

    table_id = request.POST.get('table_id')
    if table_id == 'login':
        form = LoginForm(data=request.POST, condition='login')
        if form.is_valid():
            user_input_code = form.cleaned_data.pop('code')
            code = request.session.get('image_code', "")
            if code.upper() != user_input_code.upper():
                form.add_error("code", "验证码错误")
                return render(request, 'login.html', {'form': form, 'css': loginCss})
            else:
                form.cleaned_data.pop('email')
                form.cleaned_data.pop('qq')
                admin_object = models.User.objects.filter(**form.cleaned_data).first()
                if not admin_object:
                    form.add_error("password", "用户名或密码错误")
                    return render(request, 'login.html', {'form': form, 'css': loginCss})
                else:
                    request.session["info"] = {'id': admin_object.id, 'name': admin_object.username}
                    # session可以保存7天
                    request.session.set_expiry(60 * 60 * 24 * 7)
                    # return HttpResponse('登陆成功！！！')
                    return redirect("/home/")
        else:
            return render(request, 'login.html', {'form': form, 'css': loginCss})
    elif table_id == 'signup':
        form = LoginForm(data=request.POST, condition='signup')
        if form.is_valid():
            user_input_code = form.cleaned_data.pop('code')
            code = request.session.get('image_code', "")
            if code.upper() != user_input_code.upper():
                form.add_error("code", "验证码错误")
                return render(request, 'login.html', {'form': form, 'css': signupCss})
            else:
                username_object = models.User.objects.filter(username=form.cleaned_data['username']).first()
                email_object = models.User.objects.filter(email=form.cleaned_data['email']).first()
                qq_object = models.User.objects.filter(qq=form.cleaned_data['qq']).first()
                if not username_object and not email_object and not qq_object:
                    models.User.objects.create(**form.cleaned_data)
                    return HttpResponse('OK!!!')
                print(form.cleaned_data['username'])
                if username_object:
                    form.add_error("username", "用户名已存在")
                if email_object:
                    form.add_error("email", "邮箱已存在")
                if qq_object:
                    form.add_error("qq", "qq已存在")
                return render(request, 'login.html', {'form': form, 'css': signupCss})
        else:
            return render(request, 'login.html', {'form': form, 'css': signupCss})
    elif table_id == 'forget':
        form = LoginForm(data=request.POST, condition ='forget')
        if form.is_valid():
            user_input_code = form.cleaned_data.pop('code')
            code = request.session.get('image_code', "")
            if code.upper() != user_input_code.upper():
                form.add_error("code", "验证码错误")
                return render(request, 'login.html', {'form': form, 'css': forgetCss})
            else:
                email_object = models.User.objects.filter(email=form.cleaned_data['email']).first()
                if not email_object:
                    form.add_error("email", "邮箱不存在")
                    return render(request, 'login.html', {'form': form, 'css': forgetCss})
                else:
                    return HttpResponse('密码已发送到邮件!!!')
        else:
            return render(request, 'login.html', {'form': form, 'css': forgetCss})

    # if form.is_valid():
    #     print(11231)
    #     # 验证成功，获取到的用户名和密码
    #     # {'username': 'wupeiqi', 'password': '123',"code":123}
    #     # {'username': 'wupeiqi', 'password': '5e5c3bad7eb35cba3638e145c830c35f',"code":xxx}
    #
    #     # 验证码的校验
    #     user_input_code = form.cleaned_data.pop('code')
    #     code = request.session.get('image_code', "")
    #     if code.upper() != user_input_code.upper():
    #         form.add_error("code", "验证码错误")
    #         if table_id == 'signup':
    #             return render(request, 'signup.html', {'form': form})
    #         if table_id == 'forget':
    #             return render(request, 'forget.html', {'form': form})
    #         return render(request, 'login.html', {'form': form})
    #
    #     # 去数据库校验用户名和密码是否正确，获取用户对象、None
    #     # admin_object = models.User.objects.filter(username=xxx, password=xxx).first()
    #
    #     if table_id == 'login':
    #         form.cleaned_data.pop('email')
    #         form.cleaned_data.pop('qq')
    #         admin_object = models.User.objects.filter(**form.cleaned_data).first()
    #         if not admin_object:
    #             form.add_error("password", "用户名或密码错误")
    #             # form.add_error("username", "用户名或密码错误")
    #             return render(request, 'login.html', {'form': form})
    #
    #         # 用户名和密码正确
    #         # 网站生成随机字符串; 写到用户浏览器的cookie中；在写入到session中；
    #         request.session["info"] = {'id': admin_object.id, 'name': admin_object.username}
    #         # session可以保存7天
    #         request.session.set_expiry(60 * 60 * 24 * 7)
    #         return redirect("/admin/list/")
    #     if table_id == 'signup':
    #         models.User.objects.create(**form.cleaned_data)
    #         return render(request, 'login.html', {'form': form})
    #     if table_id == 'forget':
    #         return render(request, 'login.html', {'form': form})
    # if table_id == 'signup':
    #     return render(request, 'signup.html', {'form': form})
    # elif table_id == 'forget':
    #     return render(request, 'forget.html', {'form': form})
    # else:
    #     return render(request, 'login.html', {'form': form})


def image_code(request):
    """ 生成图片验证码 """

    # 调用pillow函数，生成图片
    img, code_string = check_code()

    # 写入到自己的session中（以便于后续获取验证码再进行校验）
    request.session['image_code'] = code_string
    # 给Session设置60s超时
    request.session.set_expiry(60)

    stream = BytesIO()
    img.save(stream, 'png')
    return HttpResponse(stream.getvalue())


def logout(request):
    """ 注销 """

    request.session.clear()

    return redirect('/login/')
