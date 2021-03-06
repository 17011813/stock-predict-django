# Generated by Django 4.0.3 on 2022-06-06 06:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stock', '0007_alter_stock_name'),
    ]

    operations = [
        migrations.CreateModel(
            name='Stock_Detail',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
                ('code', models.CharField(max_length=6)),
                ('field', models.CharField(max_length=200)),
                ('products', models.CharField(max_length=200)),
                ('list_date', models.DateField()),
                ('closing_month', models.CharField(max_length=3)),
                ('chief', models.CharField(max_length=30)),
                ('pagelink', models.CharField(default='-', max_length=100)),
                ('area', models.CharField(max_length=20)),
            ],
        ),
    ]
