import asyncio
from downloader import download_audio
from analyzer import analyze_audio

async def main():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print("Testing download...")
    result = await download_audio(url)
    print("Download success:", result)
    
    print("Testing analyzer...")
    res = analyze_audio(result['audio_path'], result['video_id'])
    print("Analyzer success, got timeline length:", len(res['timeline']), "solos length:", len(res['solos']))

if __name__ == "__main__":
    asyncio.run(main())
