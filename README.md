

### 一、作品概述

#### 1、创作背景

图像识别技术在近几年有了长足的发展，如opencv，机器学习平台tensorflow、pytorch等日趋强大和完善。大家对人脸识别更是记忆深刻吧，机场，海关，支付都可以刷脸了。在这种背景下，扑克牌的图像自动识别应用似乎是小儿科的事情了。因为游戏桌台都在室内，光线充足，背景单纯，有利于图像识别算法的实施。但很高的识别准确率也并非易事，本文利用目前比较主流的yolov5目标检测算法对扑克进行识别，本文是利用在pc端训练好的权重然后对模型转换，最后部署到安卓手机上的。

打扑克牌是目前人类比较流行的娱乐方式，但人们很容易忘记打出了多少张牌和以及剩下了多少张牌，手动直接记录玩家出的牌效率很低，特别是中老年人以及记忆差的人，能够有一个自动记牌器可以提高人们对扑克牌的兴趣而且效率高，但是目前大多数的自动记牌器都是适用于线上的，例如欢乐斗地主、天天斗地主等等，很少有线下的记牌器，本人考虑到这个需求，根据具体情况制作了线下的自动记牌器的app。

#### 2、功能概要

- 注册：用户进入记牌器前要注册新用户。
- 登录：老用户可以直接用之前的密码进行登录或者注册账号来登录。
- 捐赠：通过扫描二维码付款
- 选择图片：获取权限访问手机上的图库来选择要计数的含有扑克牌的图片。
- 识别图片：点击识别图片就会调用yolov5的detect函数对图片进行检测，有cpu检测和gpu检测，根据情况选择。
- 查看：查看功能是查看你累计识别图片的扑克牌种类的数量，记录了玩家出的牌种类的数目，即玩家目前打出各种扑克牌的统计。
- 返回登录界面：退出记牌器系统，回到登陆界面。
- 清空统计数量：删除玩家之前的出牌的数量，保留当前出牌的记录。

### 二、目标群体分析

我们的目标群体以中老年人为主。老年人平时休闲时间比年轻人多大部分都喜欢打扑克牌，但是他们大多数记忆性不好打出了多少张牌很难记住，这就限制了老年人的潜力，这一步部分的人群需求更高，主要是针对这一部分群体。考虑到老年人手脚不灵活，app功能就设计的比较简单，功能一目了然，而且字体设置的就比较大一些针对一些视力差的老年人。

### 三、作品可行性分析

#### 1、技术可行性

目标检测是计算机视觉领域最重要也是最具有挑战性的分支之一。它在人们生活中得到了广泛的应用，如监控安全、自动驾驶等。目标检测的任务是定位某一类语义对象的实例。随着用于检测任务的深度学习网络的快速发展，目标检测器的性能得到了极大的提高。本文使用的yolov5就是目标检测的其中一种。YOLOv5是在YOLOv4算法的基础上做了进一步的改进，检测性能得到进一步的提升。虽然YOLOv5算法并没有与YOLOv4算法进行性能比较与分析，但是YOLOv5在COCO数据集上面的测试效果还是挺不错的。本人使用自己手机摄像头拍摄的扑克牌的做成数据集，然后利用yolov5进行两百多轮的训练，得到不错的效果，得到pt权重文件，然后将它转化为onnx模型，再转化为ncnn模型，修改网络结构，最后利用ncnn框架部署到安卓手机上，而且ncnn框架在手机端 CPU 的速度快于目前所有已知的开源框架。使得手机上运行人工智能算法成为了可能。

#### 2、经济可行性

(1) 设备费用：租用RTX3090服务器得到费用
(2) 交流电费用：使用本地电脑消耗的电费
(3) 收益：通过应用里面的捐赠获得收益

#### 3、社会可行性

本安卓手机app没有违反国家法律和政策，没有涉及到侵权，界面简洁功能简单适合老年人使用。

#### 4、前景分析

它可以帮助玩家记录各家出过的牌，自动统计出的牌，根据玩法不同。软件工作智能化，自动记牌，大大提高胜率。未来会有更多的老年人喜欢扑克牌，而有一款线下的记牌器是非常重要的。

### 四、作品设计

#### 1、开发环境描述

本人在配置环境、训练模型以及部署到安卓上的开发环境主要有：

(1) 使用的ndk版本为r22，protobuf-3.4.0，pytorch11.0,cuda11.6.134
(2) Git Bash 配置环境。
(3) VS2019: 安装nmake执行安装ncnn的命令，配置ncnn环境。
(4) Netron: 查看并且修改适合安卓移动端上运行的yolov5网络模型。
(5) Labelimg: 标注数据集。
(6) Pycharm专业版: 修改yolov5代码配置pytorch环境，连接服务器，检测模 型生成权重文件和onnx文件。
(7) Android Studio: 部署到安卓上并且添加功能。

#### 2、功能模块详细设计

本安卓app功能模块主要分为三大模块分别是登录功能、注册功能和主界面的记牌功能:

登录：登录界面下面有进入注册界面的按钮、登陆成功进入主界面和进入捐赠界面的按钮；主界面下有选择图片、用CPU识别和GPU识别、查看统计结果、清空统计结果、返回登陆界面以及捐赠按钮，app的功能模块组成如图1所示。用户打开app首先进入的是登录界面若用户没有注册过就直接登录SQLite数据库里面没有查找到该用户的信息，Login会返回false，提示登录失败。

注册：如果用户点击注册按钮通过意图跳转到注册页面。然后用户输入信息，点击注册按钮，就创建user对象存入数据，调用UserServer里面的register函数把用户信息插入到SQLite数据库里面，然后提示用户注册成功调用finish函数结束意图返回到登陆界面，用户再次登录输入信息，系统再创建一个userservice对象，调用里面的Login函数，通过查询语句，若查询到该用户信息，则返回ture否则false，若ture则进入主界面，注册登录过程如图二顺序图所示。

图像识别：进入主界面后点击选择图片申请系统访问图库权限选择图片，通过加载已经训练好的并且简化转换的param模型，调用yolov5的detect函数对图片进行识别，最后通过showObjects函数对图片标注，查看功能是通过result函数展示的，主要是对objects的label判断，然后分别将识别到的每一种扑克类别加入到bundle里面，再分别用intent的putExtra传递参数到另一个activity，然后分别用getBundleExtra和getString取出每种扑克类别的数量，最后展示出来。清空统计数据，就是把之前累加的数据置零，但会保留当前识别的数据。识别的主要原理就是利用自己修改过的yolov5模型对利用手机拍摄的三百多张扑克数据集标注然后利用租来的服务器进行训练两百多轮，然后将生成的pt权重文件转化为onnx模型，然后再对模型进行简化，得到简化后的onnx模型，再转化为ncnn便于移动端运行的框架，最后利用腾讯的开源框架ncnn方便部署到安卓上，再对代码进行修改。

#### 3、数据库设计

本系统使用的数据库管理系统是sqlite。数据库People.db的user表里面有用户的有用户名、用户密码以及性别，主要的作用是用来储存用户的注册信息，实现登录功能，如下表所示。

| 字段名 | 数据类型 | 长度 | 主键 | 描述 |
|-------|----------|------|------|------|
| Id    | integer  | 50   | 是   | 注册用户的编号 |
| username | nvarchar(20) | 20 | 否 | 用户名 |
| password | nvarchar(20) | 20 | 否 | 用户的密码 |
| sex | nvarchar(2) | 2 | 否 | 用户的性别 |

### 五、作品实现

#### 1、关键代码说明

```java
package com.tencent.yolov5ncnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class YoloV5Ncnn {
    public native boolean Init(AssetManager mgr);

    public class Obj {
        public float x;
        public float y;
        public float w;
        public float h;
        public String label;
        public float prob;
    }

    public native Obj[] Detect(Bitmap bitmap, boolean use_gpu);

    static {
        System.loadLibrary("yolov5ncnn");
    }
}

// 上面函数为图像识别提供了yolov5检测接口并且获取坐标和识别种类以及置信度,以及加载已经模型转换的权重文件，是本app的非常关键的代码

buttonImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Intent i = new Intent(Intent.ACTION_PICK);
                i.setType("image/*");
                startActivityForResult(i, SELECT_IMAGE);
            }
        });
//点击事件实现了从图库里面获取要识别的图片

Button buttonDetect = (Button) findViewById(R.id.buttonDetect);
        buttonDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                if (yourSelectedImage == null){
                    Toast.makeText(MainActivity.this, "请在相册里选择图片", Toast.LENGTH_LONG).show();
                    return;
                }


                YoloV5Ncnn.Obj[] objects = yolov5ncnn.Detect(yourSelectedImage, false);

                showObjects(objects);

            }
        });
//点击事件实现了对图片进行识别以及把目标对象框出来并显示目标名称，这是用cpu的，用GPU原理也一样。

private void showObjects(YoloV5Ncnn.Obj[] objects)
    {
        if (objects == null)
        {
            imageView.setImageBitmap(bitmap);
            return;
        }

        // draw objects on bitmap
        Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);

        final int[] colors = new int[] {
            Color.rgb( 54,  67, 244),
            Color.rgb( 99,  30, 233),
            Color.rgb(176,  39, 156),
            Color.rgb(183,  58, 103),
            Color.rgb(181,  81,  63),
            Color.rgb(243, 150,  33),
            Color.rgb(244, 169,   3),
            Color.rgb(212, 188,   0),
            Color.rgb(136, 150,   0),
            Color.rgb( 80, 175,  76),
            Color.rgb( 74, 195, 139),
            Color.rgb( 57, 220, 205),
            Color.rgb( 59, 235, 255),
            Color.rgb(  7, 193, 255),
            Color.rgb(  0, 152, 255),
            Color.rgb( 34,  87, 255),
            Color.rgb( 72,  85, 121),
            Color.rgb(158, 158, 158),
            Color.rgb(139, 125,  96)
        };

        Canvas canvas = new Canvas(rgba);

        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(4);

        Paint textbgpaint = new Paint();
        textbgpaint.setColor(Color.WHITE);
        textbgpaint.setStyle(Paint.Style.FILL);

        Paint textpaint = new Paint();
        textpaint.setColor(Color.BLACK);
        textpaint.setTextSize(26);
        textpaint.setTextAlign(Paint.Align.LEFT);

        for (int i = 0; i < objects.length; i++)
        {
            paint.setColor(colors[i % 19]);

            canvas.drawRect(objects[i].x, objects[i].y, objects[i].x + objects[i].w, objects[i].y + objects[i].h, paint);

            // draw filled text inside image
            {
                String text = objects[i].label + " = " + String.format("%.1f", objects[i].prob * 100) + "%";


//                System.out.println(objects[i].label);


                float text_width = textpaint.measureText(text);
                float text_height = - textpaint.ascent() + textpaint.descent();

                float x = objects[i].x;
                float y = objects[i].y - text_height;
                if (y < 0)
                    y = 0;
                if (x + text_width > rgba.getWidth())
                    x = rgba.getWidth() - text_width;

                canvas.drawRect(x, y, x + text_width, y + text_height, textbgpaint);

                canvas.drawText(text, x, y - textpaint.ascent(), textpaint);
            }
        }

        imageView.setImageBitmap(rgba);

}
//获取目标并把目标通过绘画工具绘画出来
public void result(View v){
        Intent intent = new Intent(MainActivity.this,second.class);
        YoloV5Ncnn.Obj[] objects = yolov5ncnn.Detect(yourSelectedImage, false);
        Bundle bundle = new Bundle();
        if (yourSelectedImage == null){
            Toast.makeText(MainActivity.this, "请在相册里选择图片", Toast.LENGTH_LONG).show();
            return;
        }else {
            sum = sum + objects.length;
            bundle.putString("sum", String.valueOf(sum));
            intent.putExtra("object",bundle);
            System.out.println(objects.length);
            for (int i = 0; i < objects.length; i++){
                if (Objects.equals(objects[i].label, "three")){
                    three++;
                    bundle.putString("three", String.valueOf(three));
                    intent.putExtra("object",bundle);
                    System.out.println(objects[i].label);
                }
                if (Objects.equals(objects[i].label, "four")){
                    four++;
                    bundle.putString("four", String.valueOf(four));
                    intent.putExtra("object",bundle);
                    System.out.println(objects[i].label);
                }
                if (Objects.equals(objects[i].label, "five")){
                    five++;
                    bundle.putString("five", String.valueOf(five));
                    intent.putExtra("object",bundle);
                    System.out.println(objects[i].label);
                }
                if (Objects.equals(objects[i].label, "six")){
                    six++;
                    bundle.putString("six", String.valueOf(six));
                    intent.putExtra("object",bundle);
                    System.out.println(objects[i].label);
                }
                if (Objects.equals(objects[i].label, "seven")){
                    seven++;
                    bundle.putString("seven", String.valueOf(seven));
                    intent.putExtra("object",bundle);
                    System.out.println(objects[i].label);
                }
                if (Objects.equals(objects[i].label, "eight")){
                    eight++;
                    bundle.putString("eight", String.valueOf(eight));
                    intent.putExtra("object",bundle);
                    System.out.println(objects[i].label);
                }
                if (Objects.equals(objects[i].label, "nine")){
                    nine++;
                    bundle.putString("nine", String.valueOf(nine));
                    intent.putExtra("object",bundle);
                    System.out.println(objects[i].label);
                }
                if (Objects.equals(objects[i].label, "ten")){
                    ten++;
                    bundle.putString("ten", String.valueOf(ten));
                    intent.putExtra("object",bundle);
                    System.out.println(objects[i].label);
                }
                if (Objects.equals(objects[i].label, "jack")){
                    jack++;
                    bundle.putString("jack", String.valueOf(jack));
                    intent.putExtra("object",bundle);
                    System.out.println(objects[i].label);
                }
                if (Objects.equals(objects[i].label, "queen")){
                    queen++;
                    bundle.putString("queen", String.valueOf(queen));
                    intent.putExtra("object",bundle);
                    System.out.println(objects[i].label);
                }
                if (Objects.equals(objects[i].label, "king")){
                    king++;
                    bundle.putString("king", String.valueOf(king));
                    intent.putExtra("object",bundle);
                    System.out.println(objects[i].label);
                }
                if (Objects.equals(objects[i].label, "ace")){
                    ace++;
                    bundle.putString("ace", String.valueOf(ace));
                    intent.putExtra("object",bundle);
                    System.out.println(objects[i].label);
                }
                if (Objects.equals(objects[i].label, "two")){
                    two++;
                    bundle.putString("two", String.valueOf(two));
                    intent.putExtra("object",bundle);
                    System.out.println(objects[i].label);
                }


            }
        }



        startActivity(intent);
    }
//对识别到的目标的数量进行统计计数，并把统计数量展示出来

public void clear(View v){
         sum = 0;
         ten = 0;
         eight = 0;
         three = 0;
         ace = 0;
         two = 0;
         jack = 0;
         nine = 0;
         queen = 0;
         seven = 0;
         five = 0;
         six = 0;
         four = 0;
         king = 0;
    }
//简单的清零操作
package com.tencent.yolov5ncnn;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

public class Myhelper extends SQLiteOpenHelper {
    public Myhelper(Context context){
        super(context,"People.db",null,2);
    }

    public void onCreate(SQLiteDatabase db){
        String sql="create Table user(id integer primary key autoincrement,username varchar(20),password varchar(20),sex varchar(2))";
        db.execSQL(sql);
    }
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {

    }
}
//连接数据库的操作实现创建数据库和建表功能
register.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                String name=username.getText().toString().trim();
                String pass=password.getText().toString().trim();

                String sexstr=((RadioButton)Register.this.findViewById(sex.getCheckedRadioButtonId())).getText().toString();
                Log.i("TAG",name+"_"+pass+"_"+sexstr);
                UserServer uService=new UserServer(Register.this);
                User user=new User();
                user.setUsername(name);
                user.setPassword(pass);

                user.setSex(sexstr);
                uService.register(user);
                Toast.makeText(Register.this, "注册成功", Toast.LENGTH_LONG).show();
                Intent intent = new Intent(Register.this,Login.class);
                startActivity(intent);
            }
        });


//注册的点击事件，通过uService.register(user);把user对象传入到uservice的register函数里面，如何获取用户名和密码，通过sql语句，实现对数据库进行插入操作。
login.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String name=username.getText().toString();
                System.out.println(name);
                String pass=password.getText().toString();
                System.out.println(pass);

                Log.i("TAG",name+"_"+pass);
                UserServer uService=new UserServer(Login.this);
                boolean flag=uService.login(name, pass);

                if(flag){
                    Log.i("TAG","登录成功");
                    Toast.makeText(Login.this, "登录成功", Toast.LENGTH_LONG).show();
//                    Intent intent = new Intent(Login.this, Register.class);
//                    startActivity(intent);
                    Intent intent = new Intent(Login.this,MainActivity.class);
                    startActivity(intent);
                }else{
                    Log.i("TAG","登录失败");
                    Toast.makeText(Login.this, "登录失败", Toast.LENGTH_LONG).show();
                }
            }
        });
//同理，利用uService.login(name, pass);将用户名和密码传入到uservice的login中，执行sql查询语句，如果能够查询到则返回ture否则返回false，然后执行相关的代码。
public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.second);


        tv0 = (TextView) findViewById(R.id.tv0);
        tv1 = (TextView) findViewById(R.id.tv1);
        tv2 = (TextView) findViewById(R.id.tv2);
        tv3 = (TextView) findViewById(R.id.tv3);
        tv4 = (TextView) findViewById(R.id.tv4);
        tv5 = (TextView) findViewById(R.id.tv5);
        tv6 = (TextView) findViewById(R.id.tv6);
        tv7 = (TextView) findViewById(R.id.tv7);
        tv8 = (TextView) findViewById(R.id.tv8);
        tv9 = (TextView) findViewById(R.id.tv9);
        tv10 = (TextView) findViewById(R.id.tv10);
        tv11 = (TextView) findViewById(R.id.tv11);
        tv12 = (TextView) findViewById(R.id.tv12);
        tv13 = (TextView) findViewById(R.id.tv13);

        Intent intent = getIntent();
        Bundle puke = intent.getBundleExtra("object");
        tv0.setText(puke.getString("sum"));
        tv1.setText(puke.getString("ace"));
        tv2.setText(puke.getString("two"));
        tv3.setText(puke.getString("three"));
        tv4.setText(puke.getString("four"));
        tv5.setText(puke.getString("five"));
        tv6.setText(puke.getString("six"));
        tv7.setText(puke.getString("seven"));
        tv8.setText(puke.getString("eight"));
        tv9.setText(puke.getString("nine"));
        tv10.setText(puke.getString("ten"));
        tv11.setText(puke.getString("jack"));
        tv12.setText(puke.getString("queen"));
        tv13.setText(puke.getString("king"));
}
//获取对应textview的id然后利用意图对另一个activity传来的数据放进到对应的textview。

public class User implements Serializable{
    private int id;
    private String username;
    private String password;

    private String sex;
    public User() {
        super();
        // TODO Auto-generated constructor stub
    }
    public User(String username, String password, String sex) {
        super();
        this.username = username;
        this.password = password;

        this.sex = sex;
    }
    public int getId() {
        return id;
    }
    public void setId(int id) {
        this.id = id;
    }
    public String getUsername() {
        return username;
    }
    public void setUsername(String username) {
        this.username = username;
    }
    public String getPassword() {
        return password;
    }
    public void setPassword(String password) {
        this.password = password;
    }
    public String getSex() {
        return sex;
    }
    public void setSex(String sex) {
        this.sex = sex;
    }
    @Override
    public String toString() {
        return "User [id=" + id + ", username=" + username + ", password="
                + password +  ", sex=" + sex + "]";
    }

}
//用户的实体类，方便对用户数据进行处理。

public boolean login(String username,String password){
        SQLiteDatabase sdb=dbHelper.getReadableDatabase();
        String sql="select * from user where username=? and password=?";
        Cursor cursor=sdb.rawQuery(sql, new String[]{username,password});
        if(cursor.moveToFirst()==true){
            cursor.close();
            return true;
        }
        return false;
    }
    public boolean register(User user){
        SQLiteDatabase sdb=dbHelper.getReadableDatabase();
        String sql="insert into user(username,password,sex) values(?,?,?)";
        Object obj[]={user.getUsername(),user.getPassword(),user.getSex()};
        sdb.execSQL(sql, obj);
        return true;
}
//用户实体的操作类，分别定义了注册函数和登录函数，实现了用户信息的插入和查询功能。
```

#### 2、作品截图


（此处应有顺序图，但未在文本中提供）

### 六、 作品设计、实现难点分析
1、难点分析
- 场景不鲁棒
由于训练样本局限导致在场景变化下会错误辨别
- 难以部署
yolov5难以部署到Android上，网络结构不一样
- 模型结构复杂
yolov5模型结构过于复杂 直接部署Android上，会导致应用崩溃，需要简化

- 设备限制
训练模型时需要消耗大CPU GPU资源，本身的设备条件难以带动

2、解决方案
- 场景不鲁棒 
通过增加图片多样性并且收集更多的数据、产生更多的数据、对数据做缩放、对数据做变换、特征选择、重新定义问题，多次训练来解决场景不鲁棒的问题。
- 难以部署
通过修改pc端网络结构删除slice层，使得该网络结构与Android端的网络结构相同，从而使PC端的yolov5模型成功部署到Android上。
- 模型结构复杂   
通过对pc端网络结构进行修改，简化训练时的网络结构，可以降低训练出来的模型部署到Android上的负载，从而解决因为网络负载过大，部署后应用崩溃的问题。
- 设备限制
由于我们自己的机器显卡不够好，在通过GPU训练学习模型时，消耗的时间较长，所以我们购买了云服务器，在上面训练模型可以有更高的效率。

