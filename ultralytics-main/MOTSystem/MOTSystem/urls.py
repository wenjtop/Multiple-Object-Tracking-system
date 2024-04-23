"""MOTSystem URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path
from django.views.static import serve
from django.conf import settings

from app.views import login, home, timeTracking, data, user, changeModel
urlpatterns = [
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}, name='media'),
    # path('admin/', admin.site.urls),

    # 登陆
    path('', login.login),
    path('login/', login.login),
    path('logout/', login.logout),
    path('image/code/', login.image_code),

    # home
    path('home/', home.home),

    # changeModel
    path('changeModel/', changeModel.changeModel),

    # 数据管理
    path('data/', data.management),
    path('data/addVideo/', data.addVideo),
    path('data/<int:nid>/delete/', data.video_delete),
    path('data/<int:nid>/videoTracking/', data.videoTracking),
    path('data/<int:nid>/download_video/', data.download_video, name='download_video'),
    path('data/<int:nid>/report/', data.report),

    # 实时跟踪
    path('timeTracking/', timeTracking.Tracking, name='process_frame'),

    # 个人信息
    path('user/', user.userInf),
    path('user/edit/', user.userEdit),
    path('user/delete/', user.userDel),

]
