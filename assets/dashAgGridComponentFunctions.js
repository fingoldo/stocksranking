var dagcomponentfuncs = (window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {});

dagcomponentfuncs.OKxLink = function (props) {
    return React.createElement('a', {
        target: '_blank', rel: 'noopener noreferrer',
        href: 'https://www.okx.com/trade-spot/' + props.value.toLowerCase(),
    },
        [React.createElement(
            'img',
            { style: { height: '20px' }, src: 'https://www.okx.com/cdn/oksupport/asset/currency/icon/' + props.value.split('-')[0].toLowerCase() + '.png?x-oss-process=image/format,webp' }
        ),
        React.createElement(
            'a',
            {
                target: '_blank', rel: 'noopener noreferrer',
                href: 'https://www.tradingview.com/chart/?symbol=OKX%3A' + props.value.replace('-', '').toUpperCase(),
                style: { paddingLeft: '6px' },
            },
            props.value
        )
        ]);
};
