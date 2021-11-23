#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QMessageBox>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pb_find_clicked();

    void on_pb_open_clicked();

    void on_pb_play_clicked();

    void on_pb_close_clicked();

    void on_pb_quit_clicked();

    void on_pb_f_area_clicked();

    void on_pb_l_area_clicked();

    void on_pb_c_area_clicked();

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
