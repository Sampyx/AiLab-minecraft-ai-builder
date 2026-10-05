const fs = require('fs');
const mineflayer = require('mineflayer');

// 1. Read the AI's build plan (using your dummy file for now)
const buildPlan = JSON.parse(fs.readFileSync('./build_plan.json', 'utf8'));

// 2. Connect the bot to your local server
const bot = mineflayer.createBot({
  host: 'localhost',
  port: 25565,
  username: 'AI_Builder'
});

bot.on('spawn', async () => {
  console.log('Bot successfully joined the server!');
  bot.chat('Hello! I am ready to build the AI structure.');
  
  // Wait 2 seconds before building
  await new Promise(resolve => setTimeout(resolve, 2000));

  // Determine a starting position (2 blocks away from where the bot spawned)
  const startPos = bot.entity.position.floored().offset(2, 0, 2);

  // 3. Loop through the JSON array and place each block
  for (let i = 0; i < buildPlan.length; i++) {
    const item = buildPlan[i];
    
    // Calculate the exact world coordinate for this specific block
    const targetX = startPos.x + item.x;
    const targetY = startPos.y + item.y;
    const targetZ = startPos.z + item.z;

    // Command the server to place the block using /setblock
    bot.chat(`/setblock ${targetX} ${targetY} ${targetZ} minecraft:${item.block}`);
    console.log(`Placed ${item.block} at ${targetX}, ${targetY}, ${targetZ}`);
    
    // Add a tiny delay (50ms) between blocks so the server doesn't kick the bot for spamming
    await new Promise(resolve => setTimeout(resolve, 50));
  }

  bot.chat('Construction complete!');
});

// Log any errors so you can debug them
bot.on('error', (err) => console.log('Bot Error:', err));
bot.on('kicked', (reason) => console.log('Bot Kicked:', reason));