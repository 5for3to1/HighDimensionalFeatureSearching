#include "mainwindow.h"
#include "ui_mainwindow.h"

#include<shlobj.h>             //Qt中执行exe文件
#include<io.h>                 //查询文件是否存在
#include<fstream>

#include <QLibrary>
#include <QDebug>
#include <QFileDialog>
#include <QLabel>
#include <QProcess>
#include <QMessageBox>
#include <QStandardItemModel>
#include <iostream>


using namespace std;

extern "C" void Topk(int);


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    choosenFileName="";                     //初始化选中的待测人脸文件名为空串

    //ui->chooseFace->setStyleSheet("background:blue");

    this->setAutoFillBackground(true);

	QPalette palette;
	palette.setBrush(QPalette::Background, QBrush(QPixmap("G:/MyQt/untitled/heaven.jpg")));
	//this->setPalette(palette);

    //setStyleSheet("QMainWindow{background-image/border-image:url(G:/MyQt/untitled/theWorld.jpg)}");
    //this->setWindowTitle("");
    //this->setWindowTitle("大规模人脸检索系统");
    this->setWindowIcon(QIcon("G:/MyQt/untitled/hust.jpg"));

	/*********************************/

	QStandardItemModel  *model = new QStandardItemModel();
	model->setRowCount(2);
//	model->setHeaderData(0, Qt::Horizontal, QString("人脸照"));

	//model->setHeaderData(1, Qt::Horizontal, QString("相似度"));

	this->ui->myTableView->setModel(model);


	//读取人脸库文件
	red_face_lib();

}

MainWindow::~MainWindow()
{
    delete ui;
	delete[] pathSimilarity;
	revoke();
}

void MainWindow::on_chooseFace_clicked()
{
    choosenFileName = QFileDialog::getOpenFileName(this,tr("选择人脸图片"),"G:/MyQt/face/test_face/",tr("Images (*.png *.bmp *.jpg *.tif *.gif)"));
    if(choosenFileName.isEmpty())
    {
        return;
    }
    qDebug()<<choosenFileName;
    //返回选择图片的索引
	ifstream input(ROOT_PATH + "face/test_face/success_image.txt");
    if (!input.is_open())
        {
            qDebug() << "file open failure" << endl;
        }
        else
        {
            face_index=0;
            string tem;
            while (!input.eof() )
            {
                input >> tem;
                if (tem==choosenFileName.toStdString())
                {
                    break;
                }
                face_index++;
            }
        }
    input.close();
    qDebug()<<face_index;

	QPixmap image;
	image.load(choosenFileName);

    //将文件选择器选择的图片显示在标签中
    ui->choosenFace->setPixmap(image);
    ui->choosenFace->setScaledContents(true);
    ui->choosenFace->resize(ui->choosenFace->height()*image.width()/image.height()
                            ,ui->choosenFace->height());

	//读取所选人脸的特征到显存中
	read_face_test(face_index);
}

void MainWindow::on_search_clicked()
{
	//删除Top-4.txt
	/*
	if (!access("G:/MyQt/Top4.txt", 0))    //参数0 表示查询是否存在
	{
		if (remove("G:/MyQt/Top4.txt") == 0)
		{
			qDebug() << "delete Top4.txt success!";
		}
		else
		{
			qDebug() << "delete Top4.txt fail!";
		}
	}
	*/

    //动态链接库Qt
	/*
	//QLibrary * mylib = NULL;
    //mylib = new QLibrary("G:/MyQt/untitled/DLLGenerator.dll");
    //mylib = new QLibrary("DLLGenerator.dll");
	QLibrary mylib("DLLGenerator.dll");
	mylib.load();
	if (!mylib.isLoaded())
	{
        qDebug() << "load dll error";
		return;
	}
	int *parm = (int *)mylib.resolve("index");
	*parm = face_index;
	DLLFUNC dllFun = (DLLFUNC)mylib.resolve("Topk");
	if (dllFun)
	{
		dllFun();										//执行showHelloCuda函数
	}
	else
	{
		printf("Can not find the function in dll!");	//可能由于函数名错误
	}
	mylib.unload();										//动态地卸载CUDAdlltest.dll
	*/

	//动态链接库Windows
	/*
	HINSTANCE hcudaDll = LoadLibrary((LPCWSTR)__T("DLLGenerator.dll"));				//动态地加载CUDAdlltest.dll
	//HINSTANCE hcudaDll = LoadLibraryA("DLLGenerator.dll");					//动态地加载CUDAdlltest.dll
	if (hcudaDll)
	{
		//DLLFUNC dllFun = (DLLFUNC)GetProcAddress(hcudaDll, "showHelloCuda");//获得函数指针
		int *parm = (int *)GetProcAddress(hcudaDll, "index");
		*parm = face_index;

		DLLFUNC dllFun = (DLLFUNC)GetProcAddress(hcudaDll, "Topk");
		if (dllFun)
		{
			dllFun();			//执行showHelloCuda函数
			//cout << endl << *parm << endl;
		}
		else
		{
			printf("Can not find the function in dll!");					//可能由于函数名错误
		}
		FreeLibrary(hcudaDll);												//动态地卸载CUDAdlltest.dll
	}
	else
	{
		printf("Load dll fail!");
	}
	*/

	//运行exe文件,设置参数
	/*
	QStringList param;
	param << QString::number(face_index, 10);
	qDebug() << param.at(0);
    QProcess::startDetached("G:\\at_HUST\\daily_HUST\\17-4\\13\\FeatureSearching\\Debug\\FeatureSearching.exe", param);
	*/
	
	//显示提示窗口
	/*
	QMessageBox message(QMessageBox::NoIcon, "Tip", "Searching face...");
	message.setIconPixmap(QPixmap("G:/MyQt/untitled/word.jpg").scaled(137, 77, Qt::KeepAspectRatio));
	//message.exec();
	message.show();
	*/

	//执行Topk，返回结果
	pathSimilarity = calc();
    
    QPixmap image;
    //图片显示在标签1中
	image.load(QString::fromStdString(pathSimilarity[0].path));
    ui->result_1->setPixmap(image);
    ui->result_1->setScaledContents(true);
    ui->result_1->resize(ui->result_1->height()*image.width()/image.height()
                            ,ui->result_1->height());
	ui->similarity_1->setText(QString("%1").arg(pathSimilarity[0].similarity));

    //图片显示在标签2中
	image.load(QString::fromStdString(pathSimilarity[1].path));
    ui->result_2->setPixmap(image);
    ui->result_2->setScaledContents(true);
    ui->result_2->resize(ui->result_2->height()*image.width()/image.height()
                            ,ui->result_2->height());
	ui->similarity_2->setText(QString("%1").arg(pathSimilarity[1].similarity));

    //图片显示在标签3中
	image.load(QString::fromStdString(pathSimilarity[2].path));
    ui->result_3->setPixmap(image);
    ui->result_3->setScaledContents(true);
    ui->result_3->resize(ui->result_3->height()*image.width()/image.height()
                            ,ui->result_3->height());
	ui->similarity_3->setText(QString("%1").arg(pathSimilarity[2].similarity));

    //图片显示在标签3中
	image.load(QString::fromStdString(pathSimilarity[3].path));
    ui->result_4->setPixmap(image);
    ui->result_4->setScaledContents(true);
    ui->result_4->resize(ui->result_4->height()*image.width()/image.height()
                            ,ui->result_4->height());
	ui->similarity_4->setText(QString("%1").arg(pathSimilarity[3].similarity));

    //图片显示在标签3中
	image.load(QString::fromStdString(pathSimilarity[4].path));
    ui->result_5->setPixmap(image);
    ui->result_5->setScaledContents(true);
    ui->result_5->resize(ui->result_5->height()*image.width()/image.height()
                            ,ui->result_5->height());
	ui->similarity_5->setText(QString("%1").arg(pathSimilarity[4].similarity));

}
