from django.db import models

# Create your models here.
class User(models.Model):
    """ 管理员 """
    username = models.CharField(verbose_name="用户名", max_length=32)
    password = models.CharField(verbose_name="密码", max_length=64)
    email = models.CharField(verbose_name="邮件", max_length=64)
    qq = models.CharField(verbose_name="QQ", max_length=64)
    def __str__(self):
        return self.username
# User.objects.create(username="wenjtop", password="892c9508d092790a74a150432299fbf7", email="1007131354@qq.com", qq="1007131354")
# User.objects.filter(username="wenjtop").update(password="91df51e16cbf1a1daf8561f6b17859d8")

class video(models.Model):
    """ video """
    videoname = models.CharField(verbose_name="视频名称", max_length=32)
    video = models.FileField(verbose_name="原视频", max_length=256, upload_to='video/')
    videoMOT = models.CharField(verbose_name="多目标跟踪视频", max_length=256, default='')
    username = models.CharField(verbose_name="用户名", max_length=32)

class obj(models.Model):
    videoID = models.ForeignKey(verbose_name="视频ID", to="video", to_field="id", on_delete=models.CASCADE)
    trackID = models.CharField(verbose_name="轨迹", max_length=32, default='')
    obj1 = models.CharField(verbose_name="目标1", max_length=128, default='')
    obj2 = models.CharField(verbose_name="目标2", max_length=128, default='')
    obj3 = models.CharField(verbose_name="目标3", max_length=128, default='')
    obj4 = models.CharField(verbose_name="目标4", max_length=128, default='')
    obj5 = models.CharField(verbose_name="目标5", max_length=128, default='')

