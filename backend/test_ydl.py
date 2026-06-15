import yt_dlp
ydl_opts = {
    'format': 'bestaudio/best',
    'extractor_args': {'youtube': {'player_client': ['ios']}},
    'http_headers': {
        'User-Agent': 'com.google.ios.youtube/19.29.1 (iPhone16,2; U; CPU iOS 17_5_1 like Mac OS X;)'
    },
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
    }],
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=rIbMbXjbW98'])
