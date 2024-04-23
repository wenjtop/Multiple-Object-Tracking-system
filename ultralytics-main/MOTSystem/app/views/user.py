from django.shortcuts import render, HttpResponse, redirect
from django.core.validators import RegexValidator
from app import models
from django import forms

from app.utils.bootstrap import BootStrapForm

class userForm(BootStrapForm):
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

def userInf(request):

    queryset = models.User.objects.filter(username=request.session["info"]["name"]).first()
    return render(request, 'userInf.html', {'queryset':queryset})

def userDel(request):
    models.User.objects.filter(username=request.session["info"]["name"]).delete()
    return render(request, 'login.html')


def userEdit(request):
    """ 编辑用户 """
    row_object = models.User.objects.filter(username=request.session["info"]["name"]).first()
    if request.method == "GET":
        # 根据ID去数据库获取要编辑的那一行数据（对象）
        dit = {'username': row_object.username, 'password':row_object.password,'email': row_object.email, 'qq':row_object.qq}
        form = userForm(initial=dit)
        return render(request, 'userEdit.html', {'form': form})

    form = userForm(data=request.POST)
    if form.is_valid():
        # 默认保存的是用户输入的所有数据，如果想要再用户输入以外增加一点值
        # form.instance.字段名 = 值
        username_object = models.User.objects.filter(username=form.cleaned_data['username']).first()
        email_object = models.User.objects.filter(email=form.cleaned_data['email']).first()
        qq_object = models.User.objects.filter(qq=form.cleaned_data['qq']).first()
        if (not username_object or username_object.id == row_object.id) and \
            (not email_object or email_object.id == row_object.id) and \
                (not qq_object or qq_object.id == row_object.id):
            models.User.objects.update(**form.cleaned_data)
            return redirect('/user/')
        if username_object and username_object.id != row_object.id:
            form.add_error("username", "用户名已存在")
        if email_object and email_object.id != row_object.id:
            form.add_error("email", "邮箱已存在")
        if qq_object and qq_object.id != row_object.id:
            form.add_error("qq", "qq已存在")
    return render(request, 'userEdit.html', {"form": form})