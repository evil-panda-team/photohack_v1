#!/usr/bin/python3
import logging
import sys
import json
import os
import urllib.request
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

TOKEN = '818151668:AAEtmcjKLfvuInsdSbDegcEMkB9Qfq34tes'
REQUEST_KWARGS = {
    # 'proxy_url': 'socks5://13.95.197.15:1080' #socks5://{0}:{1}'.format(proxy, port)
    # Optional, if you need authentication:
    # 'urllib3_proxy_kwargs': {
    #     'username': 'PROXY_USER',
    #     'password': 'PROXY_PASS',
    # }
}

result_path = "bot/static/res/"


def work(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="I hear you, wait a little please")
    msg = update.message.text
    userid = update.message.from_user.id
    selfie = bot.getUserProfilePhotos(userid, limit=1)
    fid = selfie.photos[0][0].file_id
    lastsize = selfie.photos[0][0].width
    # get biggest size
    for sizes in selfie.photos[0]:
        if sizes.width > lastsize:
            fid = sizes.file_id
            lastsize = sizes.width

    img = bot.getFile(fid)
    # saving selfie
    urllib.request.urlretrieve(img.file_path, './pics/' + fid + '.jpg')

    import common

    out_path = result_path + str(userid)

    try:
        common.fcn("pics/" + fid + '.jpg', msg, result_path=out_path)
    except Exception as e:
        print("ERROR: ", e)
        bot.send_message(chat_id=update.message.chat_id,
                         text="Something failed, ping admins, i will not respond on this command")
    else:
        results = [fn for fn in os.listdir(result_path + str(userid))]
        for res_img in results:
            img_path = out_path + "/" + res_img
            if res_img.endswith(".gif"):
                bot.send_animation(chat_id=update.message.chat_id, animation=open(img_path, 'rb'), timeout=50)
            else:
                bot.send_photo(chat_id=update.message.chat_id, photo=open(img_path, 'rb'))
        bot.send_message(chat_id=update.message.chat_id, text="Thats all results!")


def run_bot():
    updater = Updater(TOKEN, request_kwargs=REQUEST_KWARGS)
    dispatcher = updater.dispatcher

    def start(bot, update):
        bot.send_message(chat_id=update.message.chat_id, text="I'm a super bot, write to me smth!")

    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)

    echo_handler = MessageHandler(Filters.text, work)
    dispatcher.add_handler(echo_handler)

    updater.start_polling()
    updater.idle()
    updater.stop()
    print("telegram bot done")


if __name__ == "__main__":
    run_bot()
