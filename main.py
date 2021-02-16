from Updater import Updater
import os
from ShoeDetector import ShoeDetector
from FeatureExtractor import FeatureExtractor
import config
from Indexer import Indexer
from Matcher import Matcher
from QualityChecker import QualityChecker
from FeatureExtractionException import FeatureExtractionException


def imageHandler(bot, message, chat_id, local_filename):
	bot.sendMessage(chat_id, "Hi, I'm processing your request")
	is_good_quality = QualityChecker.is_good_quality(Indexer.load_image(local_filename, im_size=config.QUALITYCHECKER_IMSIZE))
	if not is_good_quality:
		bot.sendMessage(chat_id, "Your image is of a poor quality. Please, send me a better one")
	else:
		is_shoe = ShoeDetector.classify_image(Indexer.load_image(local_filename, im_size=config.CLASSIFIER_IM_SIZE))
		if not is_shoe:
			bot.sendMessage(chat_id, "Ops! Something went wrong... Make sure your image contains a shoe")
		else:
			try:
				most_similar = Matcher.get_most_similar(Indexer.load_image(local_filename))
				retrieved_images = Matcher.retrieve_items(most_similar)
				bot.sendMessage(chat_id, "These are the most similar shoes I've found")
				for im in retrieved_images:
					bot.sendImage(chat_id, config.DATASET_PATH + im, "")
			except FeatureExtractionException:
				bot.sendMessage(chat_id, "I couldn't process your photo. Please, send me a better one")


def init():
	bot_id = '1437569240:AAEd2sZ0faC1EwPvQGJPPW4xf7ohP1hTzV8'
	updater = Updater(bot_id)
	updater.setPhotoHandler(imageHandler)

	QualityChecker.init()
	ShoeDetector.init()
	FeatureExtractor.init()
	data_structure = Indexer.build_data_structure(config.DATASET_PATH)
	Matcher.init(data_structure)

	print("Bot is running...")
	updater.start()


if __name__ == "__main__":
	init()
